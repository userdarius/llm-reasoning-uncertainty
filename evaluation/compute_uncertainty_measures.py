"""Compute uncertainty measures after generating answers."""

import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader.data_utils import load_ds
from evaluation import p_true as p_true_utils
from evaluation.p_ik import get_p_ik
from evaluation.semantic_entropy import (
    EntailmentDeberta,
    EntailmentGPT4,
    EntailmentGPT4Turbo,
    EntailmentGPT35,
    EntailmentLlama,
    cluster_assignment_entropy,
    context_entails_response,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)
from techniques.speculative_decoding import (
    ModelWithProphetWrapper,
    speculative_decoding_with_prophet_model,
)
from utils import utils
from .analyze_results import analyze_run

utils.setup_logger()

EXP_DETAILS = "experiment_details.pkl"


def main(args):

    # Load tokenizer and models
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    primary_model = AutoModelForCausalLM.from_pretrained(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Initialize prophet model - for simplicity, using ;
    prophet_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-1B").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Wrap both models with the ModelWithProphetWrapper for speculative decoding
    model_and_prophet = ModelWithProphetWrapper(
        model=primary_model,
        prophet=prophet_model,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Setup wandb, directories, etc
    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ["USER"]
    scratch_dir = os.getenv("SCRATCH_DIR", ".")
    wandb_dir = f"{scratch_dir}/{user}/uncertainty"
    slurm_jobid = os.getenv(
        "SLURM_JOB_ID", None
    )  # might need to get this from rcp cluster
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"

    # Initialize wandb run: if new id, get old run and copy configs, else reuse active wandb id
    if args.assign_new_wandb_id:
        logging.info("Assign new wandb_id.")
        api = wandb.Api()
        old_run = api.run(
            f"{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}"
        )
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f"slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}",
            # For convenience, keep any 'generate_answers' configs from old run,
            # but overwrite the rest!
            # NOTE: This means any special configs affecting this script must be
            # called again when calling this script!
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir
            )

            class Restored:
                name = f"{wandb.run.dir}/{filename}"

            return Restored

    else:
        logging.info("Reuse active wandb id.")

        def restore(filename):
            class Restored:
                name = f"{wandb.run.dir}/{filename}"

            return Restored

    # Handles loading of datasets based on whether there is a distribution shift
    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid,
            args.eval_wandb_runid,
        )

        is_ood_eval = True  # pylint: disable=invalid-name
        api = wandb.Api()
        old_run_train = api.run(
            f"{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}"
        )
        filename = "train_generations.pkl"
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir
        )
        with open(f"{wandb.run.dir}/{filename}", "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config["dataset"]}, allow_val_change=True
        )
    else:
        is_ood_eval = False  # pylint: disable=invalid-name
        if args.compute_p_ik or args.compute_p_ik_answerable:
            train_generations_pickle = restore("train_generations.pkl")
            with open(train_generations_pickle.name, "rb") as infile:
                train_generations = pickle.load(infile)

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info("Beginning loading for entailment model.")
        if args.entailment_model == "deberta":
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == "gpt-4":
            entailment_model = EntailmentGPT4(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif args.entailment_model == "gpt-3.5":
            entailment_model = EntailmentGPT35(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif args.entailment_model == "gpt-4-turbo":
            entailment_model = EntailmentGPT4Turbo(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif "llama" in args.entailment_model.lower():
            entailment_model = EntailmentLlama(
                args.entailment_cache_id,
                args.entailment_cache_only,
                args.entailment_model,
            )
        else:
            raise ValueError
        logging.info("Entailment model loading complete.")

    if args.compute_p_true_in_compute_stage:
        entailment_model = None
        # This is usually not called.
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp["args"])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp["args"].dataset,
            add_options=old_exp["args"].use_mc_options,
            seed=args.random_seed,
        )
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = (
            p_true_utils.construct_few_shot_prompt(
                model=pt_model,
                dataset=pt_train_dataset,
                indices=old_exp["p_true_indices"],
                prompt=old_exp["prompt"],
                brief=old_exp["BRIEF"],
                brief_always=old_exp["args"].brief_always
                and old_exp["args"].enable_brief,
                make_prompt=utils.get_make_prompt(old_exp["args"]),
                num_generations=num_gen,
                metric=utils.get_metric(old_exp["args"].metric),
            )
        )
        del p_true_responses
        wandb.config.update({"p_true_num_fewshot": len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info("Generated few-shot prompt for p_true.")
        logging.info(80 * "#")
        logging.info("p_true_few_shot_prompt: %s", p_true_few_shot_prompt)
        logging.info(80 * "#")

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning(
            "Recompute accuracy enabled. This does not apply to precomputed p_true!"
        )
        metric = utils.get_metric(args.metric)

    # Restore outputs from `generate_answrs.py` run.
    result_dict_pickle = restore("uncertainty_measures.pkl")
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict["semantic_ids"] = []

    validation_generations_pickle = restore("validation_generations.pkl")
    with open(validation_generations_pickle.name, "rb") as infile:
        validation_generations = pickle.load(infile)

    # Setup variables for tracking entropy and embeddings.
    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation["reference"]["answers"]["text"]) > 0

    # Loop over data points and compute validation embeddings and entropies using speculative decoding
    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        question = example["question"]
        context = example["context"]
        prompt = example["prompt"]
        full_responses = example["responses"]
        most_likely_answer = example["most_likely_answer"]

        # Use speculative decoding instead of base decoding to generate responses
        sampled_responses, acceptance_ratio = speculative_decoding_with_prophet_model(
            net=model_and_prophet,
            prompt=prompt,
            seq_len=args.generate_length,
            gamma=args.speculative_gamma,
            temperature=args.temperature,
            filter_thres=args.filter_threshold,
        )

        # Compute validation embeddings and entropies using speculative decoded responses
        validation_is_true.append(most_likely_answer["accuracy"])
        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer["embedding"])

        if args.compute_predictive_entropy:
            # Token log likelihoods from speculative decoding
            log_liks = [resp.log_probabilities for resp in sampled_responses]

            # handle cases where sampled_responses or log_liks might be empty due to probabilistic sampling of speculative decoding
            if log_liks:
                log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
                entropies["regular_entropy"].append(predictive_entropy(log_liks_agg))
            # Continue with other entropy calculations
            else:
                logging.warning("Speculative decoding returned empty log likelihoods.")

            # If context entails response is required, calculate it using speculative decoded responses
            if args.compute_context_entails_response:
                entropies["context_entails_response"].append(
                    context_entails_response(
                        context, sampled_responses, entailment_model
                    )
                )

            # Compute semantic IDs for clustering, if applicable
            semantic_ids = get_semantic_ids(
                sampled_responses,
                model=entailment_model,
                strict_entailment=args.strict_entailment,
                example=example,
            )

            result_dict["semantic_ids"].append(semantic_ids)

            # Cluster assignment entropy for semantic IDs
            entropies["cluster_assignment_entropy"].append(
                cluster_assignment_entropy(semantic_ids)
            )

            # Length normalization and entropy computation
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
            entropies["regular_entropy"].append(predictive_entropy(log_liks_agg))

            # Semantic entropy calculation
            log_likelihood_per_semantic_id = logsumexp_by_id(
                semantic_ids, log_liks_agg, agg="sum_normalized"
            )
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies["semantic_entropy"].append(pe)

            # Logging speculative decoding outputs
            logging.info("Speculative Decoding Acceptance Ratio: %f", acceptance_ratio)
            logging.info(
                "semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s",
                semantic_ids,
                log_liks_agg,
                entropies,
            )

        # Further p_true computation if enabled
        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model,
                question,
                most_likely_answer["response"],
                sampled_responses,
                p_true_few_shot_prompt,
                hint=old_exp["args"].p_true_hint,
            )
            p_trues.append(p_true)
            logging.info("p_true: %s", np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info("Breaking out of main loop.")
            break

    # Additional processing, logging, and saving metrics
    logging.info("Accuracy on original task: %f", np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict["validation_is_false"] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict["validation_unanswerable"] = validation_unanswerable
    logging.info(
        "Unanswerable prop on validation: %f", np.mean(validation_unanswerable)
    )

    if "uncertainty_measures" not in result_dict:
        result_dict["uncertainty_measures"] = dict()

    if args.compute_predictive_entropy:
        result_dict["uncertainty_measures"].update(entropies)

    # Save results and any additional outputs
    utils.save(result_dict, "uncertainty_measures.pkl")

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * "#X")
        logging.info("STARTING `analyze_run`!")
        analyze_run(wandb.run.id)
        logging.info(50 * "#X")
        logging.info("FINISHED `analyze_run`!")


if __name__ == "__main__":
    parser = utils.get_parser(stages=["compute"])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f"Unkown args: {unknown}")

    logging.info("Args: %s", args)

    main(args)
