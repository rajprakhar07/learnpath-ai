jsdom UI checks (logic and DOM only; they do not test visual appearance or a real browser's PDF viewer).
    cd tests/ui && npm init -y && npm i jsdom
    python run.py &            # in the project root, with an empty data dir:  DATA_DIR=/tmp/lp python run.py
    node ui_tutor.js && node ui_practice_mastery.js && node ui_review_loop.js && node ui_multiformat.js
Each script expects a FRESH database (they load the sample files themselves) and the server on http://localhost:5000.
