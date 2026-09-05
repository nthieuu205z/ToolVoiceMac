from fastapi import APIRouter

from pipeline.languages import available_languages

router = APIRouter()


@router.get("/api/languages")
def list_languages() -> list[dict]:
    return [
        {
            "code": item.code,
            "display_name": item.display_name,
            "english_name": item.english_name,
            "video_dubbing": True,
            "text_to_voice": True,
        }
        for item in available_languages()
    ]
