#!/usr/bin/env python3
"""
App légère : retrouve une chanson à partir de quelques paroles.

Index principal = embeddings audio ELUS (text → audio sur pistes non vues).
Whisper / fichiers .txt = optionnels, pas nécessaires.
"""

import argparse
from pathlib import Path

import gradio as gr

from playlist_index import PlaylistIndex, load_elus


INDEX = None
ARGS = None


def do_index(playlist_dir, use_whisper, progress=gr.Progress()):
    global INDEX
    if not playlist_dir or not Path(playlist_dir).is_dir():
        return "Dossier playlist invalide.", None
    progress(0, desc="Chargement modèle ELUS…")
    model, symbol_to_id = load_elus(ARGS.checkpoint, ARGS.vocab, ARGS.device)
    INDEX = PlaylistIndex(model, symbol_to_id, ARGS.device)

    def cb(p, msg):
        progress(p, desc=msg)

    n = INDEX.build(
        playlist_dir,
        use_whisper=bool(use_whisper),
        whisper_model=ARGS.whisper_model if use_whisper else None,
        max_audio_sec=ARGS.max_audio_sec,
        progress_cb=cb,
    )
    cache = ARGS.index_cache
    INDEX.save(cache)
    return f"{n} pistes indexées (audio ELUS) → {cache}", _results_table([])


def do_load_cache():
    global INDEX
    if not Path(ARGS.index_cache).is_file():
        return "Aucun cache. Indexe d'abord la playlist.", None
    model, symbol_to_id = load_elus(ARGS.checkpoint, ARGS.vocab, ARGS.device)
    INDEX = PlaylistIndex(model, symbol_to_id, ARGS.device)
    n = INDEX.load(ARGS.index_cache)
    return f"Cache chargé : {n} pistes", _results_table([])


def _results_table(results):
    if not results:
        return []
    return [
        [r["rank"], f"{r['score']:.3f}", r["title"], r["snippet"], r["path"]]
        for r in results
    ]


def do_search(query, topk, mode):
    global INDEX
    if INDEX is None or not INDEX.items:
        return "Indexe d'abord ta playlist (ou charge le cache).", [], None
    if not query or not query.strip():
        return "Écris quelques paroles…", [], None
    results = INDEX.search(query.strip(), topk=int(topk), mode=mode)
    if not results:
        return "Aucun résultat (vocabulaire / requête trop courte).", [], None
    msg = f"Top {len(results)} pour : « {query.strip()} » [{mode}]"
    audio_path = results[0]["path"] if results else None
    return msg, _results_table(results), audio_path


def on_select(evt: gr.SelectData, table):
    if table is None or evt.index is None:
        return None
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row < 0 or row >= len(table):
        return None
    path = table[row][4]
    return path if path and Path(path).is_file() else None


def build_ui():
    with gr.Blocks(title="ELUS Playlist Finder") as demo:
        gr.Markdown(
            """
            # Retrouve ta chanson (ELUS)
            Tu te souviens de **quelques paroles** ? On encode ta playlist en audio
            avec ELUS, puis on cherche en **text → audio** — **sans Whisper**.
            """
        )
        playlist = gr.Textbox(
            label="Dossier playlist",
            value=ARGS.playlist or "",
            placeholder="/chemin/vers/Musique",
        )
        with gr.Row():
            btn_index = gr.Button("Indexer (audio ELUS)", variant="primary")
            btn_cache = gr.Button("Charger le cache")
        with gr.Accordion("Options avancées", open=False):
            use_whisper = gr.Checkbox(
                label="Whisper (optionnel) — seulement si tu veux aussi un index paroles",
                value=False,
            )
        status = gr.Textbox(label="Statut", interactive=False)

        gr.Markdown("### Recherche : bribes de paroles → piste audio")
        query = gr.Textbox(
            label="Paroles dont tu te souviens",
            placeholder="ex: yankee doodle went to town",
            lines=2,
        )
        with gr.Row():
            topk = gr.Slider(1, 10, value=5, step=1, label="Nombre de résultats")
            mode = gr.Dropdown(
                choices=["audio", "hybrid", "lyrics"],
                value="audio",
                label="Mode (audio = ELUS text→audio)",
            )
            btn_search = gr.Button("Chercher", variant="primary")

        msg = gr.Textbox(label="Résultat", interactive=False)
        table = gr.Dataframe(
            headers=["#", "score", "titre", "extrait", "fichier"],
            datatype=["number", "str", "str", "str", "str"],
            label="Matches",
            interactive=False,
            wrap=True,
        )
        audio = gr.Audio(label="Écouter", type="filepath")

        btn_index.click(do_index, [playlist, use_whisper], [status, table])
        btn_cache.click(do_load_cache, [], [status, table])
        btn_search.click(do_search, [query, topk, mode], [msg, table, audio])
        table.select(on_select, [table], [audio])

        gr.Markdown(
            """
            **Fonctionnement** : chaque piste → embeddings audio ELUS (fenêtres).
            Ta requête texte est projetée dans le même espace → similarité.
            Les fichiers `.txt` / Whisper restent un bonus, pas un prérequis.
            """
        )
    return demo


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/elus_librispeech_100h_best.pt")
    parser.add_argument("--vocab", default="checkpoints/symbols_en.json")
    parser.add_argument("--playlist", default="test_playlist")
    parser.add_argument("--index_cache", default="checkpoints/playlist_index.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_audio_sec", type=float, default=90.0)
    parser.add_argument("--whisper_model", default="openai/whisper-tiny")
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    ARGS = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=ARGS.server_name,
        server_port=ARGS.server_port,
        share=ARGS.share,
    )


if __name__ == "__main__":
    main()
