qdrant-up:
	docker compose up -d qdrant
	@echo "Waiting for Qdrant to be healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' rag-souverain-qdrant 2>/dev/null)" = "healthy" ]; do sleep 1; done
	@echo "Qdrant is up: http://localhost:6333"

qdrant-down:
	docker compose down

qdrant-logs:
	docker compose logs -f qdrant