import requests
import time
import bittensor as bt
from sd_net.validators.utils.uids import get_random_uids
from sd_net.protocol import ImageGenerating, pil_image_to_base64
from template.base.validator import BaseValidatorNeuron
import random
import torch
import os
import redis
from sd_net.validators.validator_proxy import ValidatorProxy
from traceback import print_exception
from typing import List
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_LIST = os.getenv("REDIS_LIST")
REWARD_URL = os.getenv("REWARD_ENDPOINT")
PROMPT_URL = os.getenv("PROMPT_ENDPOINT")


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        # TODO(developer): Anything specific to your use case you can do here
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        self.validator_proxy = ValidatorProxy(self.metagraph, self.dendrite, 8000, authentication_tokens=["test_token"])
    
    def get_prompt(self, seed: int) -> str:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        data = {
            "prompt": "an image of",
            "seed": seed,
            "max_length": 77,
            "additional_params": {},
        }

        response = requests.post(PROMPT_URL, headers=headers, json=data)
        prompt = response.json()["prompt"]
        return prompt

    def get_reward(
        self,
        miner_response: ImageGenerating,
        prompt: str,
        seed: int,
        additional_params: dict = {},
    ):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        miner_images = miner_response.images
        data = {
            "prompt": prompt,
            "seed": seed,
            "images": miner_images,
            "additional_params": additional_params,
        }
        response = requests.post(REWARD_URL, headers=headers, json=data)
        print(response)
        reward = response.json()["reward"]
        return reward

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        seed = random.randint(0, 1000)
        item = self.redis_client.blpop(REDIS_LIST, timeout=0)
        requested_data = eval(item[1])
        prompt = requested_data['prompt']
        print(prompt)
        available_uids = get_random_uids(self, k=self.config.neuron.sample_size)
        print(f"UIDS: {available_uids}")
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in available_uids],
            synapse=ImageGenerating(prompt=prompt, seed=seed),
            deserialize=False,
        )
        valid_uids = []
        valid_responses = []
        for uid, response in zip(available_uids, responses):
            if response and response.images:
                valid_uids.append(uid)
                valid_responses.append(response)

        # bt.logging.info(f"Received responses: {valid_responses}")

        rewards = [
            self.get_reward(response, prompt, seed) for response in valid_responses
        ]
        rewards = torch.FloatTensor(rewards)
        bt.logging.info(f"Scored responses: {rewards}")
        self.update_scores(rewards, valid_uids)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())
                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()
        # Update metagraph for validator proxy
        self.validator_proxy.metagraph = self.metagraph
        # Always save state.
        self.save_state()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
