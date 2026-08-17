"""FastAPI application for SchemeSathi."""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import FRONTEND_DIR
from backend.models import ChatRequest, ChatResponse, SchemeCard
from backend.scheme_loader import scheme_loader
from backend.vector_store import vector_store
from backend.chat_manager import chat_manager, ChatSession
from backend.rag_pipeline import rag_pipeline
from backend.model_registry import get_available_models_info

app = FastAPI(title='SchemeSathi API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.on_event('startup')
async def startup_event():
    print('\n--- SchemeSathi Starting Up ---')
    scheme_loader.load_data()
    vector_store.init()
    print('--- Startup Complete ---\n')


@app.get('/api/models')
async def get_models():
    """Return available models and their local vs server resolution status."""
    return get_available_models_info()


@app.post('/api/chat/new')
async def create_new_session():
    session_id = chat_manager.create_session()
    return {'session_id': session_id}


@app.post('/api/chat', response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session = chat_manager.get_session(request.session_id)
    if not session:
        chat_manager.sessions[request.session_id] = ChatSession(request.session_id)
        session = chat_manager.get_session(request.session_id)

    # Store language preference
    session.language = request.language

    history = session.get_history().copy()
    session.add_message('user', request.message)

    # Run RAG pipeline with language, selected model, and optional user API key
    reply_text, cards = await rag_pipeline.process_query(
        history, request.message, language=request.language, model_id=request.model, api_key=request.api_key
    )


    session.add_message('model', reply_text)

    session_cards = [SchemeCard(**c) for c in cards]
    return ChatResponse(
        session_id=request.session_id,
        reply=reply_text,
        schemes=session_cards,
    )


@app.post('/api/chat/stream')
async def chat_stream_endpoint(request: ChatRequest):
    session = chat_manager.get_session(request.session_id)
    if not session:
        chat_manager.sessions[request.session_id] = ChatSession(request.session_id)
        session = chat_manager.get_session(request.session_id)

    # Store language preference
    session.language = request.language

    history = session.get_history().copy()
    session.add_message('user', request.message)

    async def event_generator():
        full_text = ""
        try:
            async for event in rag_pipeline.process_query_stream(
                history, request.message, language=request.language, model_id=request.model, api_key=request.api_key
            ):
                if event["type"] == "text":
                    full_text += event["content"]
                yield f"data: {json.dumps(event)}\n\n"

            # Save LLM response to history at the end of successful streaming
            if full_text:
                session.add_message('model', full_text)
        except Exception as e:
            print(f"Error in chat_stream_endpoint: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error occurred.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@app.get('/api/schemes/{slug}')
async def get_scheme_detail(slug: str):
    detail = scheme_loader.get_scheme_detail(slug)
    if not detail or not detail.get('name'):
        raise HTTPException(status_code=404, detail='Scheme not found')
    return detail


@app.get('/api/health')
async def health_check():
    vec_count = 0
    try:
        vec_count = vector_store.collection.count() if vector_store.collection else 0
    except Exception:
        pass
    return {
        'status': 'ok',
        'schemes_loaded': len(scheme_loader.all_schemes),
        'detailed_schemes': len(scheme_loader.detailed_schemes),
        'vector_count': vec_count,
    }


# Serve frontend
css_dir = os.path.join(FRONTEND_DIR, 'css')
js_dir = os.path.join(FRONTEND_DIR, 'js')

if os.path.exists(css_dir):
    app.mount('/css', StaticFiles(directory=css_dir), name='css')
if os.path.exists(js_dir):
    app.mount('/js', StaticFiles(directory=js_dir), name='js')


@app.get('/')
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {'message': 'Frontend not found.'}


def main():
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == '__main__':
    main()
