from class_fine_tuned_gpt import NextBestActionPredictor
 
# Replace the path with the actual path to your fine-tuned model
model_path = "./gpt2-fine-tuned"
 
# Initialize the predictor
action_predictor = NextBestActionPredictor(model_path)
 
# Test with custom data
customer_input = "The product I received is damaged and I want to return it."
agent_action = action_predictor.predict_action(customer_input)
 
print(f'User: {customer_input}')
print(f'Agent: {agent_action}')

