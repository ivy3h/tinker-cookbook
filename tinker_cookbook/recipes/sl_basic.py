import chz
import sys
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import asyncio

#import inspect
#print(inspect.signature(train.Config))

def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = chat_datasets.NoRobotsBuilder(common_config=common_config)
    if 1: 
        dataset = FromConversationFileBuilder(
            common_config=common_config, file_path="/srv/nlprx-lab/share6/jhe478/tinker-cookbook/data/s1k_1.1_best.jsonl"
        )

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/tinker-examples/sl_basic",
            "model_name": model_name,
            "dataset_builder": dataset,
            "learning_rate": 1e-5,
            "lr_schedule": "linear",
            "num_epochs": 5,
            "eval_every": 8,
            "adam_beta1": 0.9,                  # β1 = 0.9
            "adam_beta2": 0.95,                 # β2 = 0.95
        }
    )


def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
