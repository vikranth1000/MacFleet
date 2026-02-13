.PHONY: proto test lint format install dev clean

proto:
	python -m grpc_tools.protoc \
		-I macfleet/comm/proto \
		--python_out=macfleet/comm/proto \
		--grpc_python_out=macfleet/comm/proto \
		macfleet/comm/proto/control.proto
	# Fix relative import in generated grpc file
	sed -i '' 's/^import control_pb2/from macfleet.comm.proto import control_pb2/' \
		macfleet/comm/proto/control_pb2_grpc.py

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=macfleet --cov-report=term-missing

lint:
	ruff check macfleet/ tests/
	mypy macfleet/ --ignore-missing-imports

format:
	ruff format macfleet/ tests/
	ruff check --fix macfleet/ tests/

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
