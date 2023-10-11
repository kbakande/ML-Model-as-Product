# Makefile for the ml_pipeline_project

# Variables
  # Google Cloud Storage URL
GCS_URL = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
PREPROCESS_OUTPUT = "preprocess_output.csv"

# Command to run the preprocess method
preprocess: $(PREPROCESS_OUTPUT)

$(PREPROCESS_OUTPUT):
	@echo "Running preprocess method..."
	python -c "from src.mlpipeline import MLPipeline; \
               pipeline = MLPipeline(); \
               df = pipeline.preprocess('$(GCS_URL)'); \
               df.to_csv('$(PREPROCESS_OUTPUT)', index=False)"

# Command to run the train method using the output of the preprocess method
train: $(PREPROCESS_OUTPUT)
	@echo "Running train method..."
	python -c "import pandas as pd; \
               from src.mlpipeline import MLPipeline; \
               df = pd.read_csv('$(PREPROCESS_OUTPUT)'); \
               pipeline = MLPipeline(); \
               model, evals_result = pipeline.train(df); \
               print('Training complete.')"

# Command to run the predict method
predict:
	@echo "Running predict method..."
	python -c "from src.mlpipeline import MLPipeline; \
               pipeline = MLPipeline(); \
               predictions_df = pipeline.predict('$(GCS_URL)'); \
               print(predictions_df.head())"

# Command to run unittest cases
test:
	@echo "Running unittest cases..."
	python -m unittest tests/test_mlpipeline.py

# Command to run pytest cases
pytest:
	@echo "Running pytest cases..."
	pytest tests/pytest_mlpipeline.py

# Command to run all methods in sequence
all: preprocess train predict

# Command to display help information
help:
	@echo "Available commands:"
	@echo "  make preprocess   - Run the preprocess method"
	@echo "  make train        - Run the train method"
	@echo "  make predict      - Run the predict method"
	@echo "  make test         - Run unittest cases"
	@echo "  make pytest       - Run pytest cases"
	@echo "  make all          - Run preprocess, train, and predict methods in sequence"
	@echo "  make help         - Display this help information"

.PHONY: preprocess train predict test all help