
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
class NextBestActionPredictor:
    def __init__(self, model_path, tokenizer_path="gpt2"):
        """
        Initializes the predictor with the fine-tuned model and tokenizer.
       
        :param model_path: Path to the fine-tuned GPT-2 model.
        :param tokenizer_path: Path to the tokenizer. Defaults to "gpt2" if the same tokenizer is used.
        """
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.model.eval()  # Set the model to evaluation mode
 
    def predict_action(self, customer_input):
        """
        Predicts the next best action based on the customer's input.
       
        :param customer_input: The customer's message as a string.
        :return: Predicted next best action as a string.
        """
        example_context = "User: I want to update my address. Agent: I can help you with that. Please provide your new address."
        prompt = f"{example_context} User: {customer_input} Agent:"
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 10,  # Adjust max_length if needed
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2
        )
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, '').strip()
        return decoded_output.split("User:")[0].strip()  # Return the next best action
 
# Example usage:
# model_path = "../../resources/Next_Best_Action/gpt2-fine-tuned"
# action_predictor = NextBestActionPredictor(model_path)
# customer_input = "user order is delayed by several days and its at our delivery hub."
# agent_action = action_predictor.predict_action(customer_input)
# print(f'User: {customer_input}')
# print(f'Agent: {agent_action}')
 
