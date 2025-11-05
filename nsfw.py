from gradio_client import Client

client = Client("Heartsync/NSFW-Uncensored-photo")
result = client.predict(
		prompt="Hello!!",
		negative_prompt="text, watermark, signature, cartoon, anime, illustration, painting, drawing, low quality, blurry",
		seed=0,
		randomize_seed=True,
		width=1024,
		height=1024,
		guidance_scale=7,
		num_inference_steps=28,
		api_name="/infer"
)
print(result)