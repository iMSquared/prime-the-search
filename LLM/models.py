import json
import os
import time
from copy import deepcopy
from typing import Dict, List, Union
from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types.text_generation import TextGenerationOutput
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential
from termcolor import colored
from transformers import AutoTokenizer

from LLM.utils.misc import find_project_base_dir
from LLM.utils.text_typings import Sequence, Token
from Simulation.pybullet.custom_logger import LLOG
import concurrent.futures

logger = LLOG.get_logger()

class LMFailure(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f'{self.message}'



class LMClient:
    def __init__(self, config) -> None:
        ## NOTE (SJ): We now need to input the actual prompt config instead of whole config 
        self.BASE_PATH = find_project_base_dir()
        self.prompt_config = config["project_params"]["overridable"]["prompt_params"]['policy']

        self.ip = self.prompt_config['ip']
        self.model_name = self.prompt_config['model']
        print(f"Model name: {self.model_name}")

        if "gpt" in self.model_name:
            self.tokenize, self.client = self.set_openai()
            self.inference_route = self.llm
        else:
            self.tokenizer, self.client = self.set_tgi(**self.prompt_config)
            self.inference_route = self.slm

        self.prefix: str = ""
        
    def set_openai(self, **configs):
        API_KEY_PATH = self.BASE_PATH / "LLM/api_token.json"
        if os.path.exists(API_KEY_PATH):
            with open(API_KEY_PATH, "r") as f:
                api_key = json.load(f)['api_token']
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")

        return None, OpenAI(api_key=api_key)
    
    
    def set_tgi(self, **configs):
        tokenizer = AutoTokenizer.from_pretrained(configs['model'])
        client = InferenceClient(
                model=configs['ip'],
                headers={
                    "Content-Type": "application/json",
                },
            )
        return tokenizer, client
    

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(40))
    def slm(self, message, params):
        assert False, "Not implemented"
        r"""
        "parameters": {
            "best_of": 1,
            "decoder_input_details": true,
            "details": true,
            "do_sample": true,
            "max_new_tokens": 20,
            "repetition_penalty": 1.03,
            "return_full_text": false,
            "seed": null,
            "stop": ["photographer"],
            "temperature": 0.5,
            "top_k": 10,
            "top_n_tokens": 5,
            "top_p": 0.95,
            "truncate": null,
            "typical_p": 0.95,
            "watermark": true
        }
        """
        # Set up the parameters for the API call
        params = {
            "prompt": message,
            'details': self.prompt_config['logprobs'],
            'do_sample': self.prompt_config['sampling'],
            "max_new_tokens": self.prompt_config['max_tokens'],
            "temperature": self.prompt_config['temperature'] if self.prompt_config['temperature'] != 0 else 0.1,
            'stop_sequences': [',','}'],
        }

        # Add any additional generation arguments

        batch_data = [params for _ in range(self.prompt_config['num_return'])]

        def gen_text(data):
            response = self.client.text_generation(**data)
            return response

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.prompt_config['num_return']) as executor:
            outputs = list(executor.map(gen_text, batch_data))

        return outputs

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
    def llm(self, message, params):
        r'''
        Models: https://platform.openai.com/docs/models/continuous-model-upgrades
        Params: https://platform.openai.com/docs/api-reference/chat/create
        '''
        response = self.client.chat.completions.create(
                                                    messages=message,
                                                    model=self.model_name,
                                                    logprobs=params.get('log_probs', self.prompt_config['logprobs']),
                                                    max_tokens = params.get('max_tokens', self.prompt_config['max_tokens']),
                                                    n = params.get('num_return', self.prompt_config['num_return']),
                                                    temperature = params.get('temperature', self.prompt_config['temperature'] if self.prompt_config['temperature'] != 0 else 0.1),
                                                )
                      
        return response
            
    def process_input(self, _input_message: Union[str, List[Dict[str, str]]], add_prefix=False, return_string=False):
        r"""
            args:
                input_message: str or list of dictionaries
                add_prefix: bool prefix for guiding the LM
                return_string: bool
        """

        input_message = deepcopy(_input_message)
        if isinstance(input_message, str):
            conversation = [{"role": "user", "content": input_message}]
        elif isinstance(input_message, list):
            assert input_message[-1]['role'] == 'user', f'Input message must end with user but {input_message} was given'
            conversation = input_message
        else:
            raise NotImplementedError('Input message must be either a string or a list of dictionaries')
        

        if add_prefix:
            self.prefix = '{"'+self.prompt_config['generation_requirements'][0]+'": "'
            conversation.append({"role": "assistant", "content": self.prefix})
        else:
            self.prefix = ""


        if return_string:
            conversation = self.tokenizer.apply_chat_template(conversation=conversation, tokenize=False)[:-5]
        
        return conversation
    

    def create_char_to_token_map(self, tokens):
        map_char_to_token = {}
        token_index = 0
        char_index = 0

        for token in tokens:
            for _ in token:
                map_char_to_token[char_index] = token_index
                char_index += 1
            token_index += 1
        return map_char_to_token
        
    def process_api_response(self, response: Union[ChatCompletion, List[TextGenerationOutput]]) -> List[Sequence]:

        standard_response = []
        if isinstance(response, ChatCompletion):
            for idx, choice in enumerate(response.choices):
                # tokens = [(t.token, t.logprob, t.bytes) for t in choice.logprobs.content] 
                # sequence = Sequence(tokens={idx: Token(text=token[0], log_prob= token[1], index=idx, bytes=len(token[2])) for idx, token in enumerate(tokens)}, 
                #                     text=choice.message.content, 
                #                     char_to_token_map=self.create_char_to_token_map([t[0] for t in tokens]))

                sequence = Sequence(text=choice.message.content,)
                standard_response.append(sequence)

        elif isinstance(response[-1], TextGenerationOutput):
            for idx, TextGenerationOutput in enumerate(response):
                tokens = [t for t in TextGenerationOutput.details.tokens]
                sequence = Sequence(tokens={idx: Token(text=token.text, log_prob= token.logprob, index=idx, bytes=len(list(token.text.encode("utf-8")))) for idx, token in enumerate(tokens)}, 
                                    text=TextGenerationOutput.generated_text,
                                    char_to_token_map=self.create_char_to_token_map([t.text for t in tokens]))
                standard_response.append(sequence)
        else:
            raise NotImplementedError(f'Response type of {type(response[-1])} not supported')
            

        return standard_response


    def get_response(self, prompt, generation_args, add_prefix=False, )-> List[Sequence]:
        i = 0
        response = None
        status, status_message = 'error', 'not started'
        # input_prompt = self.process_input(prompt, add_prefix=add_prefix, return_string=True)
        input_prompt = prompt
        processed_response = None

        while True:
            if status == 'success': break
            if i > 5: break
            i += 1
            ## Step 1: Generate response from LM
            start_time = time.time()

            response = self.inference_route(input_prompt, generation_args)
            end_time = time.time() - start_time                  
            logger.info(f"Time taken for LM inference: {end_time}")
            processed_response = self.process_api_response(response)

            status = 'success'

        

        if status != 'success':
            logger.warning(f"LM failed to give a valid response")
            return None
    
        return processed_response