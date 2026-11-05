# AI Virtual Mouse

Sterowanie myszą gestami dłoni z kamery: MediaPipe (śledzenie dłoni) → maszyna stanów gestów →
pynput (wstrzykiwanie zdarzeń myszy/klawiatury do systemu). Ruch, lewy/prawy/podwójny klik,
przeciąganie, scroll w pionie i poziomie, zoom (Ctrl + kółko), sprzęgło.

## Szybki start

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python main.py --dry-run         # najpierw bez sterowania myszą: sprawdź gesty w oknie HUD
```

```bash
python main.py                   # właściwe uruchomienie
```

Model `hand_landmarker.task` (~8 MB) pobiera się automatycznie przy pierwszym uruchomieniu do
`models/`.

**Wyjście awaryjne:** `Ctrl+Alt+Q` (działa globalnie, także bez fokusu na oknie i z `--no-hud`),
pauza `Ctrl+Alt+P`. W oknie HUD: `q`/`ESC` wyjście, `p` pauza. Przy każdym wyjściu, także po
`Ctrl+C` i po błędzie, wszystkie wciśnięte przyciski i klawisze są zwalniane.

## Gesty

| Akcja | Gest | Zabezpieczenie |
| --- | --- | --- |
| Ruch kursora | otwarta dłoń | kotwica na kłykciach (LM 5+9), nie na czubku palca; filtr 1€ |
| Lewy klik | krótkie zbliżenie **kciuk + wskazujący** (< 0,25 s) | kursor zamrożony już przy zbliżaniu palców, więc klik trafia tam, gdzie celowałeś |
| Przeciąganie | przytrzymanie **kciuk + wskazujący** (> 0,25 s) albo ruch > 40 px przy złączonych palcach | rozłączenie palców = upuszczenie; krótka utrata dłoni (< 0,2 s) nie przerywa przeciągania |
| Prawy klik | **kciuk + środkowy** | jeden klik na jedno zbliżenie + cooldown 0,4 s |
| Podwójny klik | **kciuk + serdeczny** (albo dwa szybkie lewe kliki) | cooldown 0,5 s |
| Scroll | **wskazujący + środkowy** wyprostowane i złączone, serdeczny + mały zgięte; przesuń dłoń w górę/dół/bok | działa jak joystick: odległość od punktu startu ustala *prędkość*; strefa martwa |
| Zoom | „rogi”: **wskazujący + mały** w górze, środkowy + serdeczny zgięte; rozsuwaj/zsuwaj kciuk i wskazujący | Ctrl + kółko, skokowo (zakres 0,12 dłoni na jeden ząbek) |
| Sprzęgło | **pięść** | `absolute`: pauza wyjścia; `--relative`: „podniesienie myszy” |

Szczypnięcia rób jak gest „OK” (pozostałe palce wyprostowane). Szczypnięcie z zaciśniętymi
pozostałymi palcami wygląda dla systemu jak pięść.

Każdy pinch jest „uzbrajany” dopiero, gdy palce były rozwarte. Dzięki temu wyjście z zoomu ze
złączonymi palcami albo dłoń, która wraca do kadru w trakcie szczypania, nie wywołują
przypadkowego kliknięcia.

## Opcje

| Flaga | Znaczenie |
| --- | --- |
| `--camera-id N` | numer kamery (domyślnie 0) |
| `--no-hud` | bez okna podglądu (mniej CPU) |
| `--sensitivity X` | wzmocnienie kursora; `1.5` = o 1/3 mniejszy ruch ręki na ten sam ruch kursora |
| `--relative` | tryb względny jak prawdziwa mysz (pięść = podniesienie myszy) |
| `--dry-run` | rozpoznawanie gestów bez dotykania prawdziwej myszy |
| `--width/--height/--fps` | żądany tryb kamery (domyślnie 1280×720@60, kamera może dać mniej) |
| `--screen WxH` | ręczny rozmiar ekranu, gdy autodetekcja zawiedzie |
| `--model PATH`, `--no-mirror`, `--no-hotkeys`, `--log-level` | jak w nazwie |

Pozostałe parametry (progi, filtr, ROI, prędkość scrolla) są w `config.py`.
