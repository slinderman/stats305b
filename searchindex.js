Search.setIndex({"docnames": ["assignments/hw0/hw0", "assignments/hw1/hw1", "index", "lectures/01_distributions", "lectures/02_contingency_tables", "lectures/03_logreg", "lectures/99_references"], "filenames": ["assignments/hw0/hw0.ipynb", "assignments/hw1/hw1.ipynb", "index.md", "lectures/01_distributions.ipynb", "lectures/02_contingency_tables.md", "lectures/03_logreg.md", "lectures/99_references.md"], "titles": ["HW0: PyTorch Primer", "HW1: Logistic Regression", "Overview", "Discrete Distributions and the Basics of Statistical Inference", "Contingency Tables", "Logistic Regression", "References"], "terms": {"stats305b": [0, 1, 3], "stanford": [0, 2], "univers": [0, 2], "winter": [0, 2], "2024": [0, 2, 3], "your": [0, 5], "name": [0, 3], "collabor": 0, "hour": [0, 2, 3], "spent": 0, "pleas": [0, 1, 3], "let": [0, 1, 3, 4, 5], "us": [0, 1, 3, 4, 5, 6], "know": [0, 3, 4, 5], "how": [0, 3, 4, 5], "mani": [0, 3, 5], "total": [0, 3, 4, 5], "you": [0, 1, 2, 3, 4, 5], "thi": [0, 1, 2, 3, 4, 5], "assign": [0, 3], "so": [0, 2, 3, 4, 5], "we": [0, 1, 2, 3, 4, 5], "can": [0, 1, 3, 4, 5], "calibr": 0, "futur": 0, "feedback": 0, "alwai": [0, 3], "welcom": [0, 2], "ll": [0, 1, 2, 3, 4, 5], "python": [0, 2, 3], "cours": [0, 1, 2, 3, 4, 5], "lab": 0, "help": [0, 2], "get": [0, 1, 2, 3, 4], "up": [0, 2, 3], "speed": [0, 2], "It": [0, 3, 4, 5], "introduc": [0, 3, 4], "s": [0, 2, 3], "equival": [0, 3, 4, 5], "numpi": 0, "arrai": 0, "bell": 0, "whistl": 0, "run": [0, 1, 3, 5], "gpu": 0, "support": [0, 1, 3], "automat": [0, 1], "differenti": [0, 1, 3, 5], "If": [0, 1, 3, 4, 5], "re": [0, 1, 3], "come": [0, 1, 5], "from": [0, 1, 2, 3, 4, 5], "matlab": 0, "probabl": [0, 1, 2, 3, 4, 5, 6], "avoid": 0, "costli": 0, "loop": 0, "comput": [0, 1, 3, 4], "over": [0, 3, 5], "dimens": 0, "an": [0, 1, 2, 3, 4, 5], "here": [0, 1, 3, 4, 5], "trick": 0, "ha": [0, 3, 5], "excel": 0, "librari": 0, "sampl": [0, 3], "evalu": [0, 3, 4], "log": [0, 1, 3], "much": [0, 3, 5], "import": [0, 1, 3, 4], "dist": 0, "matplotlib": [0, 1, 3], "pyplot": [0, 1, 3], "plt": [0, 1, 3], "ar": [0, 1, 3, 4, 5], "The": [0, 2, 3, 4, 5], "document": 0, "alreadi": [0, 5], "great": 0, "tutori": 0, "rather": [0, 3, 4], "than": [0, 2, 3, 4], "recreat": 0, "wheel": 0, "start": [0, 1, 3, 4, 5], "read": [0, 2], "onc": 0, "ve": [0, 1, 3], "through": 0, "try": [0, 3, 5], "function": [0, 3, 4, 5, 6], "like": [0, 2, 3, 4, 5], "arang": [0, 3], "reshap": 0, "etc": [0, 4, 5], "follow": [0, 1, 2, 3, 4, 5], "0": [0, 2, 3, 4, 5], "8": [0, 1, 2, 3], "note": [0, 1, 3, 4], "For": [0, 3, 4, 5], "ones": 0, "below": [0, 1, 4, 5], "don": [0, 4, 5], "t": [0, 1, 3, 4, 5], "liter": 0, "specifi": [0, 3, 5], "list": [0, 1], "code": [0, 1, 2], "sequenc": 0, "repeat": 0, "doe": [0, 1, 4, 5], "arbitrari": [0, 5], "number": [0, 1, 3, 5], "life": [0, 3], "easier": 0, "faster": [0, 5], "hang": 0, "x": [0, 1, 3, 4, 5], "where": [0, 1, 3, 4, 5], "i": [0, 1, 2, 3, 4, 5], "j": [0, 3, 4, 5], "sum": [0, 3, 4, 5], "two": [0, 1, 3], "dimension": [0, 3], "exampl": 0, "distanc": 0, "matrix": [0, 1, 4], "d": [0, 3, 5, 6], "euclidean": 0, "between": [0, 3, 4, 5], "10": [0, 1, 2, 3, 4, 6], "dtype": [0, 3], "float": 0, "answer": [0, 1], "should": [0, 2, 3, 4, 5], "0000": 0, "8284": 0, "6569": 0, "4853": 0, "11": [0, 2, 3, 4], "3137": 0, "extract": 0, "submatrix": 0, "row": [0, 1, 3, 4], "column": [0, 1, 3, 4], "A": [0, 1, 4, 6], "25": [0, 3], "14": [0, 1, 2, 3], "15": [0, 2, 4], "16": 0, "19": [0, 2, 3], "creat": 0, "binari": [0, 4, 5, 6], "mask": 0, "m": [0, 2, 4], "same": [0, 1, 3, 4, 5], "true": [0, 1, 3, 4], "onli": [0, 1, 3, 4, 5], "divis": [0, 1, 3], "fals": [0, 1, 3], "add": [0, 3, 5], "one": [0, 1, 2, 3, 4], "entri": [0, 3], "after": [0, 5], "updat": [0, 3, 5], "place": [0, 4], "9": [0, 1, 2, 3, 4], "12": [0, 2], "13": [0, 1, 2, 3], "17": [0, 2], "18": 0, "20": [0, 2, 3], "22": [0, 2, 3], "23": [0, 4, 6], "24": [0, 2, 6], "doc": [0, 3], "object": [0, 5], "fit": [0, 1], "poisson": 0, "mixtur": 0, "model": [0, 2, 4, 5, 6], "draw": [0, 2, 3, 4], "50": 0, "rate": [0, 1, 3], "One": [0, 4, 5], "awesom": 0, "thing": [0, 5], "about": [0, 1, 2, 3, 4], "thei": [0, 2, 3, 4, 5], "too": [0, 3, 5], "p": [0, 1, 3, 4, 5, 6], "equal": [0, 2, 3, 4, 5], "mathrm": [0, 1, 3, 4, 5], "poi": 0, "lambda": [0, 3, 4], "ldot": [0, 1, 3, 4, 5], "3679": 0, "1839": 0, "0613": 0, "0153": 0, "1353": 0, "2707": 0, "1804": 0, "0902": 0, "0498": 0, "1494": 0, "2240": 0, "1680": 0, "0183": 0, "0733": 0, "1465": 0, "1954": 0, "point": [0, 1, 3, 5], "under": [0, 1, 3, 4, 5], "gamma": [0, 1, 5, 6], "aka": [0, 5], "concentr": 0, "invers": [0, 3, 4, 5], "scale": [0, 3, 4], "0336": 0, "5905": 0, "0540": 0, "1000": [0, 3], "begin": [0, 1, 3, 4, 5], "align": [0, 1, 3, 4, 5], "frac": [0, 1, 3, 4, 5], "end": [0, 1, 2, 3, 4, 5], "hist": [0, 3], "normal": [0, 4], "histogram": [0, 1, 3], "data": [0, 2, 3, 4, 5, 6], "batch": 0, "100": [0, 5], "independ": [0, 1, 3], "standard": [0, 3, 4, 5], "random": [0, 3, 4, 5], "variabl": [0, 2, 3, 4, 5, 6], "varianc": [0, 3, 4, 5], "mean": [0, 1, 2, 3, 5], "process": [0, 1, 3], "remov": 0, "bug": 0, "program": [0, 3], "must": [0, 4], "put": [0, 5], "them": [0, 2, 4], "edsger": 0, "dijkstra": 0, "skill": 0, "appli": [0, 2, 5], "statist": [0, 2, 4, 6], "hone": 0, "class": [0, 1, 2, 3, 5], "exercis": 0, "tool": [0, 1, 4], "techniqu": [0, 2, 5], "In": [0, 1, 2, 3, 4, 5], "particular": [0, 4, 5], "focu": [0, 2, 3, 4], "give": [0, 1, 3, 5], "some": [0, 3, 4, 5], "directli": [0, 4], "interrog": 0, "what": [0, 1, 2, 3, 4, 5], "happen": 0, "snippet": 0, "hopefulli": 0, "learn": [0, 2], "understand": [0, 4], "go": [0, 5], "given": [0, 3, 4, 5], "replac": 0, "feeling": 0, "powerless": 0, "when": [0, 1, 3, 4, 5], "encount": 0, "inscrut": 0, "confid": 0, "masteri": 0, "knowledg": 0, "effect": [0, 1, 5], "aris": 0, "error": [0, 3, 4, 5], "grammar": 0, "wai": [0, 1, 2], "make": [0, 1, 3], "sens": [0, 1], "base": [0, 3, 4], "english": 0, "sai": [0, 3, 4, 5], "enjoi": 0, "beauti": 0, "park": 0, "becaus": [0, 2], "have": [0, 1, 2, 3, 4, 5], "adject": 0, "verb": 0, "belong": 0, "similar": [0, 3], "evalaut": 0, "express": [0, 3, 4, 5], "oper": 0, "differ": [0, 1, 3, 4, 5], "action": 0, "int": [0, 3, 4], "unfamiliar": 0, "write": [0, 1, 3], "instead": [0, 3, 4, 5], "throw": 0, "spaghetti": 0, "wall": 0, "see": [0, 1, 3, 4, 5], "stick": 0, "better": [0, 1, 5], "ask": [0, 3], "its": [0, 3, 4, 5], "applic": 0, "well": [0, 2, 3, 4, 5], "attribut": 0, "approach": [0, 3, 4, 5], "might": [0, 4], "new": [0, 3, 4], "friend": 0, "person": [0, 4], "would": [0, 3], "randomli": 0, "guess": [0, 3], "explicitli": [0, 5], "question": [0, 1], "todo": 0, "intuit": [0, 1, 3, 5], "reason": [0, 3, 4], "why": [0, 1, 5], "other": [0, 1, 2, 3, 4, 5, 6], "call": [0, 2, 3, 4, 5], "abov": [0, 1, 3, 4, 5], "three": [0, 3], "describ": [0, 1], "notic": 0, "each": [0, 1, 2, 3, 4], "out": [0, 1, 4, 5], "method": [0, 2, 6], "behavior": [0, 3], "expect": [0, 3, 4, 5], "denot": [0, 1, 3, 4, 5], "magic": 0, "command": [0, 1], "which": [0, 1, 3, 4, 5], "open": 0, "ad": 0, "ipython": 0, "short": [0, 3], "interact": 0, "backbon": 0, "jupyt": [0, 1, 3], "notebook": [0, 1], "colab": 0, "cell": [0, 1, 4], "return": [0, 3, 5], "veri": [0, 4, 5], "next": [0, 3, 4, 5], "just": [0, 1, 2, 3, 4, 5], "shft": 0, "enter": 0, "debugg": 0, "wa": [0, 1, 3], "rais": 0, "Then": [0, 1, 2, 3, 4, 5], "investig": 0, "valu": [0, 1, 3, 4, 5], "were": [0, 3, 4], "befor": [0, 3], "line": [0, 1, 3, 5], "led": 0, "crash": 0, "navig": 0, "ipd": 0, "consol": 0, "find": [0, 1, 3, 5, 6], "both": [0, 2, 3, 4], "tensor_a": 0, "tensor_b": 0, "explain": [0, 1], "work": [0, 1, 4], "exit": 0, "quit": 0, "buggi": 0, "randn": 0, "result": [0, 1, 4], "sometim": [0, 4], "subtl": 0, "either": 0, "caus": [0, 4], "ani": [0, 1, 4, 5], "messag": 0, "situat": 0, "anywher": 0, "confirm": 0, "still": 0, "print": [0, 1, 3], "synthet": 0, "practic": [0, 3], "gener": [0, 2, 3, 4, 5, 6], "manual_se": [0, 3], "305": [0, 3], "n": [0, 1, 3, 4, 5, 6], "100_000": 0, "hstack": 0, "beta_tru": 0, "y": [0, 1, 3, 4, 5], "set": [0, 1, 3, 4, 5], "logist": [0, 6], "loss_fn": 0, "nn": 0, "mseloss": 0, "beta_hat": 0, "requires_grad": 0, "learning_r": 0, "num_iter": 0, "loss": 0, "gradient": [0, 4], "descent": [0, 6], "rang": [0, 2, 3, 5], "forward": 0, "pass": 0, "y_pred": 0, "append": 0, "item": 0, "zero": [0, 3, 5], "backward": 0, "grad": 0, "none": [0, 3], "weight": [0, 2, 5], "no_grad": 0, "final": [0, 2, 3, 4, 5], "paramet": [0, 1, 4, 5], "train": 0, "beta_fin": 0, "detach": 0, "recov": 0, "coeffici": [0, 5], "our": [0, 3, 4, 5], "check": [0, 5], "vector": [0, 1, 3, 4, 5], "close": [0, 5], "correspond": [0, 3, 4, 5], "element": [0, 3], "within": 0, "atol": 0, "do": [0, 1, 3, 4, 5], "programat": 0, "condit": [0, 3, 5], "edit": [0, 2], "margin": [0, 3, 4], "1e": 0, "elementwis": 0, "output": [0, 1, 5], "done": [0, 1], "cleanli": 0, "stand": 0, "format": [0, 1, 3], "second": [0, 3, 5], "argument": [0, 3], "boolean": 0, "modifi": 0, "descrption": 0, "easi": [0, 3], "comparison": 0, "first": [0, 2, 3, 4, 5], "lossess": 0, "clue": 0, "locat": [0, 2], "fix": [0, 3], "extrem": 0, "nice": [0, 5], "long": [0, 3], "take": [0, 1, 3, 4, 5], "wrap": [0, 1], "easiest": 0, "tqdm": [0, 1, 3], "packag": [0, 1, 3], "often": [0, 3, 4], "want": [0, 1, 3, 5], "quickli": 0, "prototyp": 0, "quick": 0, "dirti": 0, "solut": [0, 5], "doesn": 0, "need": [0, 1, 3, 4, 5], "satisfi": 0, "But": [0, 4, 5], "report": 0, "summari": 0, "experi": 0, "cme": 0, "193": 0, "introduct": [0, 6], "scientif": 0, "unit": [0, 3, 5], "meet": [0, 3], "wednesdai": [0, 2, 3], "30": [0, 1, 2, 3, 4], "quarter": 0, "strongli": [0, 5], "recommend": 0, "concurr": 0, "enrol": 0, "mit": 0, "miss": [0, 3], "semest": 0, "softwar": [0, 6], "while": [0, 3, 4], "relat": 0, "stat": [0, 2, 5], "305b": [0, 2], "review": 0, "seri": 0, "programm": 0, "exce": [0, 1], "80": [0, 1, 6], "charact": [0, 1], "width": [0, 1, 3], "editor": [0, 1], "vertic": [0, 1], "ruler": [0, 1], "exceed": [0, 1, 3], "limit": [0, 1, 3, 4, 5], "convert": [0, 1], "pdf": [0, 1, 3], "simplest": [0, 1], "option": [0, 1], "browser": [0, 1], "sure": [0, 1, 4], "aren": [0, 1, 3, 5], "cut": [0, 1], "off": [0, 1, 3], "mai": [0, 1, 2, 3], "altern": [0, 1, 3, 4, 5], "download": [0, 1], "ipynb": [0, 1], "nbconvert": [0, 1], "yourlastnam": [0, 1], "_hw": [0, 1], "renam": [0, 1], "file": [0, 1], "instal": [0, 1], "anaconda": [0, 1], "manag": [0, 1], "conda": [0, 1], "c": [0, 1, 5, 6], "upload": [0, 1], "gradescop": [0, 1], "tag": [0, 1], "correctli": [0, 1], "e": [0, 1, 3, 4, 5, 6], "all": [0, 1, 3, 4, 5], "relev": [0, 1], "section": [0, 1], "post": [0, 1, 3], "ed": [0, 1], "oh": [0, 1], "submit": [0, 1], "hw": [0, 1, 2], "algorithm": [1, 2, 3, 5], "discret": [1, 2, 4, 5], "homework": [1, 2], "ingredi": 1, "colleg": [1, 4], "footbal": [1, 4, 5], "game": [1, 3, 5], "2023": [1, 2, 3, 6], "season": [1, 3], "brade": 1, "predict": [1, 3, 5], "winner": 1, "team": [1, 3, 5], "k": [1, 3, 4, 6], "beta_k": 1, "basic": [1, 2, 4, 5], "higher": 1, "rel": [1, 4], "speak": 1, "formal": 1, "intut": 1, "odd": [1, 5], "beat": [1, 3], "beta_": 1, "index": [1, 3, 4], "h": [1, 6], "indic": [1, 3, 4], "home": [1, 3, 5], "awai": [1, 3, 5], "respect": [1, 3], "y_i": [1, 5], "whether": [1, 4, 5], "won": [1, 3], "equat": 1, "sim": [1, 3, 4, 5], "bern": [1, 3, 4, 5], "big": [1, 2], "sigma": [1, 4, 5], "cdot": [1, 3, 4], "sigmoid": [1, 5], "view": [1, 3], "covari": [1, 3, 4], "x_i": [1, 3], "mathbb": 1, "r": [1, 3], "x_": [1, 3, 4], "case": [1, 3, 4, 5], "text": [1, 3, 4, 5], "o": [1, 3], "w": [1, 3], "beta": [1, 3, 4], "fall": [1, 3, 4], "avail": 1, "github": 1, "page": [1, 6], "load": 1, "outcom": [1, 3], "individu": 1, "wrangl": 1, "feed": 1, "torch": [1, 3], "panda": [1, 3], "pd": [1, 3], "allgam": [1, 3], "read_csv": [1, 3], "http": [1, 3, 6], "raw": [1, 3], "githubusercont": [1, 3], "com": [1, 3, 6], "slinderman": [1, 3], "winter2024": [1, 3], "01_allgam": [1, 3], "csv": [1, 3], "id": [1, 3], "week": [1, 3, 5], "type": [1, 3, 6], "date": [1, 2, 3], "time": [1, 2, 3, 4, 5], "tbd": [1, 2, 3], "complet": [1, 3], "neutral": [1, 3], "site": [1, 3], "confer": [1, 3], "attend": [1, 3], "score": 1, "win": [1, 3, 5], "prob": [1, 3], "pregam": [1, 3], "elo": [1, 3], "postgam": 1, "excit": [1, 3], "highlight": [1, 3, 4], "401550883": [1, 3], "regular": [1, 2, 3, 6], "08": [1, 3], "26t17": [1, 3], "00": [1, 3, 4], "000z": [1, 3], "nan": [1, 3], "401525434": [1, 3], "26t18": [1, 3], "49000": [1, 3], "american": [1, 3, 6], "athlet": [1, 3, 4, 6], "fb": [1, 3], "001042": [1, 3], "1471": [1, 3], "1385": [1, 3], "346908": [1, 3], "401540199": [1, 3], "26t19": [1, 3], "uac": [1, 3], "fc": [1, 3], "7": [1, 2, 3, 5], "025849": [1, 3], "6": [1, 2, 3], "896909": [1, 3], "401520145": [1, 3], "26t21": [1, 3], "17982": [1, 3], "usa": [1, 3], "591999": [1, 3], "1369": [1, 3], "1370": [1, 3], "821333": [1, 3], "401525450": [1, 3], "26t23": [1, 3], "15356": [1, 3], "41": [1, 3], "760751": [1, 3], "1074": [1, 3], "1122": [1, 3], "311493": [1, 3], "401532392": [1, 3], "23867": [1, 3], "mid": [1, 3, 4, 5], "045531": [1, 3], "1482": [1, 3], "1473": [1, 3], "547378": [1, 3], "401540628": [1, 3], "patriot": [1, 3], "077483": [1, 3], "608758": [1, 3], "401520147": [1, 3], "21407": [1, 3], "mountain": [1, 3], "west": [1, 3], "28": [1, 2, 3], "819154": [1, 3], "1246": [1, 3], "1241": [1, 3], "282033": [1, 3], "401539999": [1, 3], "meac": [1, 3], "001097": [1, 3], "122344": [1, 3], "401523986": [1, 3], "27t00": [1, 3], "63411": [1, 3], "001769": [1, 3], "1462": [1, 3], "1412": [1, 3], "698730": [1, 3], "33": [1, 3, 6], "drop": [1, 3], "construct": [1, 3, 4, 5], "respons": [1, 3, 4, 5, 6], "l": [1, 5], "defin": [1, 3, 4], "sum_": [1, 3, 4, 5], "_2": [1, 5], "hyperparamet": [1, 3], "control": [1, 4], "strength": [1, 4, 5], "ell_2": 1, "distribut": [1, 2, 5], "bernoulli": [1, 4, 5], "averag": [1, 3, 5], "neg": [1, 3, 4, 5], "likelihood": [1, 4], "against": [1, 4], "obtain": [1, 3, 4, 5], "pytorch": [1, 3], "now": [1, 3, 5], "provid": [1, 5], "deliver": 1, "plot": [1, 3], "curv": 1, "brief": 1, "discuss": [1, 3, 4], "top": [1, 3, 4, 5], "rank": 1, "multipl": 1, "markdown": 1, "organ": 1, "autograd": 1, "without": [1, 5], "unless": 1, "sort": 1, "pre": 1, "singular": 1, "mathemat": [1, 5], "context": [1, 3], "hypothesi": [1, 4], "dataset": [1, 3], "empir": 1, "evid": 1, "invert": [1, 5], "initi": [1, 5], "beta_0": [1, 5], "briefli": 1, "compar": [1, 5], "converg": [1, 3], "anoth": [1, 5], "look": [1, 3], "earlier": 1, "think": [1, 2, 3, 5], "choos": [1, 3, 4], "priori": [1, 3], "chang": [1, 5], "assess": 1, "perform": [1, 3, 5], "held": [1, 3, 5], "test": 1, "includ": [1, 4, 5], "least": [1, 3, 4, 5], "analysi": [1, 2, 4, 5, 6], "conduct": 1, "assignemnt": 1, "best": [1, 2, 5], "Is": 1, "signific": 1, "justifi": [1, 3], "offici": 2, "ii": 2, "unoffici": 2, "realli": [2, 3], "cover": 2, "linear": [2, 6], "sequenti": 2, "latent": [2, 6], "autoregress": 2, "transform": 2, "On": 2, "side": [2, 3, 4], "few": [2, 3, 5], "convex": 2, "optim": [2, 5, 6], "approxim": [2, 3, 4, 5], "bayesian": [2, 6], "infer": [2, 5, 6], "mcmc": 2, "variat": 2, "concept": 2, "implement": [2, 5], "scratch": 2, "By": 2, "strong": [2, 4], "grasp": 2, "classic": [2, 3], "modern": 2, "instructor": 2, "scott": [2, 3, 6], "linderman": [2, 3], "ta": 2, "xavier": 2, "gonzalez": 2, "leda": 2, "liang": 2, "term": [2, 3, 5], "mondai": [2, 3], "1": [2, 3, 4, 5, 6], "2": [2, 3, 4, 5], "50pm": 2, "room": 2, "380": 2, "380d": 2, "offic": [2, 3], "10am": 2, "2nd": 2, "floor": 2, "loung": 2, "wu": 2, "tsai": 2, "neurosci": 2, "institut": 2, "thursdai": [2, 3], "5": [2, 3], "7pm": 2, "sequoia": 2, "hall": 2, "207": 2, "bowker": 2, "fridai": [2, 3], "3": [2, 3, 4, 5, 6], "5pm": 2, "build": [2, 3, 5], "360": 2, "361a": 2, "student": 2, "comfort": [2, 3], "undergradu": 2, "multivari": [2, 3, 4], "calculu": 2, "algebra": 2, "emphas": 2, "profici": 2, "requir": 2, "hw0": 2, "primer": 2, "part": [2, 5], "agresti": [2, 3, 6], "alan": [2, 6], "categor": [2, 4, 5, 6], "john": [2, 6], "wilei": [2, 6], "son": [2, 6], "2002": [2, 6], "link": [2, 5], "switch": 2, "research": [2, 4], "paper": 2, "chapter": [2, 5], "textbook": 2, "topic": 2, "jan": 2, "agr02": [2, 3, 6], "ch": [2, 4], "conting": [2, 3], "tabl": [2, 3], "mlk": 2, "dai": 2, "No": [2, 4], "regress": 2, "4": [2, 3, 5], "exponenti": [2, 5], "famili": [2, 5], "glm": 2, "select": [2, 3, 5], "diagnost": 2, "29": 2, "l1": 2, "fht10": [2, 6], "lss14": [2, 6], "31": 2, "probit": 2, "ac93": [2, 6], "psw13": [2, 6], "feb": 2, "mix": 2, "midterm": 2, "hidden": 2, "markov": 2, "presid": 2, "21": 2, "field": 2, "26": 2, "recurr": 2, "neural": 2, "network": 2, "attent": 2, "tranform": 2, "mar": 2, "state": 2, "space": [2, 4], "layer": 2, "s4": 2, "s5": 2, "mamba": 2, "graph": 2, "structur": 2, "denois": 2, "diffus": 2, "everyth": [2, 5], "els": 2, "There": [2, 3, 5], "due": [2, 3], "roughli": [2, 3], "everi": [2, 3], "last": [2, 3, 4, 5], "bit": [2, 5], "more": [2, 3, 4, 5], "substanti": 2, "rest": [2, 4], "releas": 2, "mon": 2, "fri": 2, "59pm": 2, "wed": 2, "bring": 2, "cheat": 2, "sheet": 2, "5x11": 2, "piec": 2, "march": 2, "30pm": 2, "percentag": 2, "particip": 2, "lectur": [3, 4], "block": 3, "throughout": 3, "common": [3, 5], "toss": 3, "bias": 3, "coin": 3, "head": [3, 4, 6], "event": 3, "flip": 3, "tail": [3, 4], "mass": 3, "pmf": [3, 4], "succinctli": 3, "mildli": 3, "overload": 3, "nomenclatur": 3, "repres": [3, 4, 5], "clear": [3, 4], "var": [3, 4, 5], "iid": [3, 4], "trial": 3, "bin": [3, 4], "Its": 3, "np": 3, "infti": [3, 4, 5], "product": [3, 4, 5], "stai": 3, "constant": [3, 5], "suppos": [3, 4, 5], "spike": 3, "fire": 3, "neuron": 3, "simplic": 3, "assum": [3, 4, 5], "divid": 3, "size": [3, 5], "proport": 3, "reals_": [3, 4, 5], "moreov": [3, 5], "shouldn": 3, "matter": 3, "determin": [3, 5], "resolut": 3, "detect": 3, "care": 3, "separ": 3, "As": 3, "po": [3, 4], "fact": 3, "properti": 3, "appropri": [3, 4, 5], "assumpt": [3, 5], "shortli": 3, "far": 3, "talk": [3, 5], "scalar": [3, 4], "extend": [3, 5], "idea": [3, 5], "count": [3, 4], "consid": [3, 4, 5], "die": 3, "face": 3, "probabilit": 3, "mbpi": [3, 4], "pi_1": 3, "pi_k": 3, "delta_": [3, 4], "left": [3, 4, 5], "sum_k": 3, "right": [3, 4, 5], "simplex": 3, "embed": [3, 5], "real": [3, 5], "roll": 3, "cat": 3, "prod_": [3, 4], "bbi": [3, 4], "otherwis": 3, "natur": [3, 4], "categori": 3, "hot": 3, "mbe_1": 3, "mbe_k": 3, "th": 3, "posit": [3, 4, 5], "mbx": [3, 4, 5], "x_k": 3, "represent": 3, "straightforward": 3, "z_i": 3, "came": 3, "mult": [3, 4], "cx_n": 3, "x_1": [3, 5], "cov": [3, 4], "bmatrix": [3, 4], "pi_2": 3, "vdot": [3, 4], "_": [3, 4, 5], "ij": [3, 4], "pi_i": [3, 4], "pi_j": 3, "neq": 3, "collect": [3, 5], "ident": [3, 5], "lambda_i": 3, "bullet": [3, 4], "lambda_k": 3, "notat": [3, 4], "render": 3, "depend": [3, 4, 5], "specif": [3, 5], "lambda_1": 3, "lambda_": [3, 4, 5], "word": [3, 5], "simpl": [3, 5], "those": [3, 4], "mbtheta": 3, "observ": [3, 4, 5], "reduc": [3, 4, 5], "cl": [3, 5], "theta": [3, 4, 5], "hat": [3, 4, 5], "mathsf": [3, 5], "arg": 3, "max": [3, 5], "singl": [3, 4, 5], "global": [3, 5], "unknown": 3, "deriv": [3, 5], "dif": [3, 4, 5], "solv": 3, "yield": [3, 4, 5], "fraction": [3, 5], "could": [3, 4, 5], "truli": 3, "star": [3, 5], "certain": [3, 4, 5], "achiev": [3, 5], "cramer": 3, "rao": 3, "lower": [3, 5], "bound": 3, "sqrt": [3, 4, 5], "cn": 3, "ci": [3, 4], "squar": [3, 4, 5], "root": 3, "diagon": 3, "partial": 3, "confus": 3, "yourself": 3, "poor": 3, "precis": [3, 5], "map": [3, 5], "cx": 3, "mapsto": [3, 5], "itself": 3, "nabla_": 3, "treat": 3, "e_": 3, "int_": 3, "mbzero": [3, 5], "again": [3, 4], "cov_": 3, "underbrac": 3, "twice": [3, 5], "hessian": [3, 5], "nabla": [3, 4, 5], "2_": [3, 4], "interestingli": [3, 4], "revisit": 3, "later": [3, 5], "null": [3, 4], "ch_0": [3, 4], "theta_0": 3, "exploit": 3, "z": [3, 4], "subscript": 3, "simplifi": [3, 5], "chi": [3, 4], "degre": [3, 4], "freedom": [3, 4], "non": [3, 4, 5], "mbtheta_0": 3, "larg": [3, 4], "canon": 3, "ratio": [3, 5], "lagrang": 3, "multipli": 3, "finit": [3, 5], "sec": 3, "detail": 3, "hypothes": [3, 4], "level": [3, 4], "alpha": [3, 4, 5], "95": [3, 4], "96": [3, 4], "pm": [3, 4], "continu": [3, 5], "found": [3, 4], "togeth": 3, "gaussian": [3, 5], "sinc": [3, 4, 5], "tempt": 3, "interpret": [3, 4], "fallaci": 3, "misinterpret": 3, "frequentist": 3, "To": [3, 5], "claim": 3, "adopt": 3, "perspect": 3, "posterior": [3, 4, 5], "prior": [3, 4, 5], "bay": 3, "rule": 3, "denomin": 3, "analogu": 3, "summar": [3, 4, 5], "captur": [3, 4], "infinit": [3, 5], "choic": [3, 5], "quantil": 3, "howev": [3, 4, 5], "inde": [3, 4], "subject": 3, "sourc": 3, "critic": 3, "noth": 3, "weak": 3, "uninform": 3, "advantag": 3, "admit": [3, 4], "most": [3, 5], "conjug": [3, 4], "densiti": 3, "b": [3, 5], "shape": 3, "uniform": 3, "propto": 3, "mode": [3, 5], "posteriori": [3, 5], "small": [3, 4], "cumul": 3, "cdf": [3, 5], "incomplet": 3, "smaller": 3, "thu": [3, 5], "approx": [3, 4, 5], "match": [3, 5], "littl": [3, 5], "tonight": 3, "michigan": 3, "wolverin": 3, "washington": 3, "huski": 3, "mod": 3, "That": [3, 4, 5], "setup": 3, "defaultdict": 3, "user": 3, "anaconda3": 3, "lib": 3, "python3": 3, "auto": 3, "py": 3, "tqdmwarn": 3, "iprogress": 3, "ipywidget": 3, "readthedoc": 3, "io": 3, "en": 3, "stabl": 3, "user_instal": 3, "html": 3, "autonotebook": 3, "notebook_tqdm": 3, "googl": 3, "spreadsheet": 3, "1lul": 3, "n2miih7ace47zagj_4wspbgrhy9e04zujmevf40": 3, "export": 3, "timestamp": 3, "net": 3, "slot": 3, "hold": [3, 4, 5], "midnight": 3, "tuesdai": 3, "03": 3, "swl1": 3, "ye": 3, "predicted_scor": 3, "tensor": [3, 4], "float32": 3, "predicted_hist": 3, "histogramdd": 3, "imshow": 3, "extent": 3, "cmap": 3, "grei": 3, "xtick": 3, "ytick": 3, "xlabel": 3, "ylabel": 3, "titl": 3, "f": [3, 5], "len": 3, "grid": 3, "colorbar": 3, "collegefootballdata": 3, "past_scor": 3, "past_hist": 3, "umdriv": 3, "01_umdriv": 3, "uwdriv": 3, "01_uwdriv": 3, "def": 3, "compute_drive_prob": 3, "even": [3, 4, 5], "though": [3, 5], "granular": 3, "encod": 3, "integ": 3, "td": 3, "fg": 3, "allow": [3, 4], "offens": 3, "defens": 3, "um_off_prob": 3, "um_off_n": 3, "um_def_prob": 3, "um_def_n": 3, "uw_off_prob": 3, "uw_off_n": 3, "uw_def_prob": 3, "uw_def_n": 3, "fig": 3, "ax": 3, "subplot": 3, "figsiz": 3, "sharex": 3, "sharei": 3, "bar": 3, "color": 3, "gold": 3, "label": 3, "purpl": 3, "set_titl": 3, "legend": 3, "set_xtick": 3, "set_ylim": 3, "set_xlabel": 3, "set_ylabel": 3, "surpris": 3, "per": 3, "me": 3, "high": [3, 4], "fly": 3, "isn": [3, 4, 5], "amaz": 3, "anywai": 3, "middl": 3, "drawn": 3, "um": 3, "uw": 3, "vice": 3, "versa": 3, "um_avg_prob": 3, "uw_avg_prob": 3, "pat": 3, "2pt": 3, "convers": 3, "98": 3, "01": [3, 4], "bunch": 3, "num_gam": 3, "num_driv": 3, "point_val": 3, "um_point": 3, "uw_point": 3, "scatter": 3, "60": [3, 4], "lw": 3, "marker": 3, "ms": 3, "xlim": 3, "ylim": 3, "gca": 3, "set_aspect": 3, "accord": 3, "upper": [3, 5], "hand": 3, "hail": 3, "victor": [3, 6], "pr": [3, 4, 5], "2f": 3, "66": 3, "spread": 3, "victori": 3, "edgecolor": 3, "axvlin": 3, "ls": 3, "also": [3, 4, 5], "bet": 3, "52": 3, "82": 3, "pro": 3, "favor": 3, "56": 3, "show": [3, 4, 5], "favorit": [3, 5], "anyth": 3, "sim_hist": 3, "column_stack": 3, "Not": 3, "bad": 3, "clearli": 3, "pick": 3, "digit": 3, "trend": 3, "appar": 3, "34": 3, "closer": 3, "suggest": 3, "fell": 3, "ala": 3, "my": 3, "jim": 3, "harbaugh": 3, "bobblehead": 3, "prize": 3, "fare": 3, "basi": 3, "01_final": 3, "um_finals_prob": 3, "um_finals_n": 3, "uw_finals_prob": 3, "uw_finals_n": 3, "avg": 3, "grai": 3, "black": 3, "vs": 3, "media": 3, "interview": 3, "player": [3, 4], "plai": [3, 4], "who": [3, 4, 5], "touchdown": 3, "outperform": 3, "fun": [3, 4], "watch": [3, 4], "hope": 3, "had": [3, 4], "tast": 3, "seem": 3, "stretch": 3, "exactli": [3, 4], "likewis": [3, 4], "procedur": 3, "These": [3, 5], "sophist": 3, "complex": [3, 4], "joint": [3, 4], "pair": [3, 5], "interest": [4, 5], "sound": 4, "lot": 4, "soon": 4, "enough": [4, 5], "turn": 4, "plenti": 4, "boil": [4, 5], "down": [4, 5], "relationship": 4, "nation": 4, "championship": 4, "analys": 4, "love": 4, "hate": 4, "increasingli": 4, "repetit": [4, 6], "injuri": 4, "sustain": 4, "devast": 4, "consequ": 4, "increas": [4, 5], "risk": 4, "chronic": 4, "traumat": 4, "encephalopathi": 4, "cte": 4, "recent": 4, "studi": 4, "mckee": [4, 6], "et": 4, "al": 4, "mma": [4, 6], "jama": [4, 6], "neurolog": [4, 6], "amateur": 4, "school": 4, "york": 4, "sad": 4, "articl": 4, "definit": [4, 5], "diagnos": 4, "via": [4, 6], "autopsi": 4, "brain": 4, "152": 4, "peopl": 4, "contact": [4, 6], "sport": [4, 6], "di": 4, "ag": 4, "variou": 4, "overdos": 4, "suicid": 4, "neurodegen": 4, "diseas": 4, "Of": 4, "92": 4, "soccer": 4, "hockei": 4, "wrestl": 4, "rugbi": 4, "63": 4, "upon": 4, "neuropatholog": [4, 6], "48": 4, "45": 4, "44": 4, "89": 4, "With": [4, 5], "associ": [4, 5, 6], "causal": 4, "caveat": 4, "pi_": 4, "1j": 4, "i1": 4, "triangleq": [4, 5], "kei": 4, "factor": [4, 5], "perp": 4, "iff": 4, "foral": 4, "homogen": 4, "usual": 4, "noisi": 4, "free": 4, "special": 4, "vec": 4, "ravel": 4, "explanatori": [4, 5], "condition": 4, "mbx_": 4, "mbpi_": 4, "mbx_i": [4, 5], "2x2": [4, 5], "hypergeom": 4, "arriv": 4, "adapt": 4, "blitzstein": [4, 6], "hwang": [4, 6], "bh19": [4, 6], "abbrevi": 4, "cumbersom": 4, "consist": 4, "ind": 4, "in0": 4, "impli": [4, 5], "substitut": 4, "binomi": 4, "cancel": 4, "group": 4, "varieti": 4, "latter": 4, "omega": 4, "recal": 4, "i0": 4, "omega_i": 4, "omega_1": 4, "omega_0": 4, "conveni": 4, "magnitud": [4, 5], "confound": 4, "2x2x2": 4, "ijk": 4, "mayb": 4, "amount": 4, "concis": 4, "2x2xk": 4, "theta_": 4, "theta_k": 4, "measur": [4, 5], "sign": 4, "mle": [4, 5], "estim": 4, "pi": [4, 5], "wald": 4, "usign": 4, "asymptot": 4, "nonlinear": [4, 5], "maximum": 4, "inform": [4, 5], "g": [4, 6], "order": [4, 5], "taylor": [4, 5], "around": [4, 5], "pmatrix": 4, "_i": 4, "_j": 4, "plug": [4, 5], "shown": 4, "accept": 4, "region": 4, "revers": 4, "impos": 4, "constraint": 4, "constrain": 4, "subset": [4, 5], "outer": [4, 5], "larger": [4, 5], "sup_": 4, "unconstrain": 4, "mu": [4, 5], "monoton": [4, 5], "lead": [4, 5], "geq": [4, 5], "min": [4, 5], "conclud": 4, "involv": 4, "unif": 4, "almost": 4, "nevertheless": [4, 5], "mont": 4, "carlo": 4, "credibl": 4, "correl": 4, "fundament": 4, "present": 4, "ultim": 4, "sever": [4, 5], "contin": 5, "featur": 5, "oppon": 5, "problem": 5, "form": 5, "took": 5, "305a": 5, "pretti": 5, "mbbeta": 5, "beta_j": 5, "x_j": 5, "ordinari": 5, "ol": 5, "wrong": 5, "issu": 5, "produc": 5, "valid": 5, "necessarili": 5, "misspecifi": 5, "violat": 5, "homoskedast": 5, "crazi": 5, "intermedi": 5, "keep": 5, "ensur": 5, "squash": 5, "particularli": 5, "attract": 5, "compon": 5, "logit": 5, "simpler": 5, "calcul": 5, "beta_1": 5, "x_p": 5, "beta_p": 5, "unfortun": 5, "unlik": 5, "maxim": 5, "minim": [5, 6], "residu": 5, "rememb": 5, "direct": 5, "steepest": 5, "tell": 5, "decreas": 5, "move": 5, "underestim": 5, "opposit": 5, "overestim": 5, "mbbeta_": 5, "leftarrow": 5, "mbbeta_t": 5, "alpha_t": 5, "step": 5, "iter": 5, "chosen": 5, "behav": 5, "alorithm": 5, "local": 5, "optimum": 5, "optima": 5, "stronger": 5, "guarante": 5, "w_i": 5, "semi": 5, "eigenvalu": 5, "quadrat": 5, "leq": 5, "smooth": 5, "norm": 5, "max_": 5, "mbu": 5, "bbs_": 5, "tfrac": 5, "sphere": 5, "meant": 5, "tight": 5, "mbbeta_0": 5, "gap": 5, "epsilon": 5, "sub": 5, "linearli": 5, "exist": 5, "lim_": 5, "satur": 5, "goe": 5, "diverg": 5, "grow": 5, "problemat": 5, "avert": 5, "textcolor": 5, "red": 5, "penal": 5, "penalti": 5, "mbi": 5, "remain": 5, "path": [5, 6], "cross": 5, "spheric": 5, "stanc": 5, "strang": 5, "entir": 5, "whole": 5, "At": 5, "minimum": 5, "stepsiz": 5, "recurs": 5, "leverag": 5, "consider": 5, "widetild": 5, "stationari": 5, "succ": 5, "vanilla": 5, "repeatedli": 5, "until": 5, "damp": 5, "improv": 5, "stabil": 5, "backtrack": 5, "search": 5, "lipschitz": 5, "doubl": 5, "incred": 5, "slowli": 5, "statement": 5, "possibl": 5, "expand": 5, "volum": 6, "792": 6, "url": 6, "onlinelibrari": 6, "doi": 6, "book": 6, "1002": 6, "0471249688": 6, "jame": 6, "albert": 6, "siddhartha": 6, "chib": 6, "polychotom": 6, "journal": 6, "669": 6, "679": 6, "1993": 6, "joseph": 6, "jessica": 6, "crc": 6, "press": 6, "2019": 6, "jerom": 6, "friedman": 6, "trevor": 6, "hasti": 6, "rob": 6, "tibshirani": 6, "coordin": 6, "2010": 6, "jason": 6, "lee": 6, "yuekai": 6, "sun": 6, "michael": 6, "saunder": 6, "proxim": 6, "ewton": 6, "composit": 6, "siam": 6, "1420": 6, "1443": 6, "2014": 6, "ann": 6, "jess": 6, "mez": 6, "bobak": 6, "abdolmohammadi": 6, "morgan": 6, "butler": 6, "bertrand": 6, "russel": 6, "huber": 6, "madelin": 6, "uretski": 6, "katharin": 6, "babcock": 6, "jonathan": 6, "cherri": 6, "alvarez": 6, "brett": 6, "martin": 6, "clinic": 6, "young": 6, "expos": 6, "impact": 6, "1037": 6, "1050": 6, "nichola": 6, "polson": 6, "windl": 6, "\u00f3": 6, "lya": 6, "108": 6, "504": 6, "1339": 6, "1349": 6, "2013": 6}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"hw0": 0, "pytorch": 0, "primer": 0, "1": [0, 1], "construct": 0, "tensor": 0, "problem": [0, 1], "2": [0, 1], "3": [0, 1], "4": [0, 1], "5": [0, 1], "6": 0, "broadcast": 0, "fanci": 0, "index": 0, "distribut": [0, 3, 4], "debug": 0, "sleuth": 0, "type": 0, "shape": 0, "dir": 0, "pdb": 0, "breakpoint": 0, "linear": [0, 5], "regress": [0, 1, 5], "assert": 0, "torch": 0, "allclos": 0, "f": 0, "string": 0, "plot": 0, "progress": 0, "bar": 0, "time": 0, "more": 0, "resourc": 0, "7": 0, "submiss": [0, 1], "instruct": [0, 1], "hw1": 1, "logist": [1, 2, 5], "The": 1, "bradlei": 1, "terri": 1, "model": [1, 3], "data": 1, "0": 1, "preprocess": 1, "loss": 1, "function": 1, "gradient": [1, 5], "descent": [1, 5], "implement": 1, "check": 1, "your": 1, "newton": [1, 5], "s": [1, 4, 5], "method": [1, 4, 5], "hessian": 1, "posit": 1, "definit": 1, "critic": 1, "revis": 1, "improv": 1, "evalu": 1, "reflect": 1, "overview": 2, "prerequisit": 2, "book": 2, "tent": 2, "schedul": 2, "assign": 2, "exam": 2, "grade": 2, "discret": 3, "basic": 3, "statist": 3, "infer": [3, 4], "bernoulli": 3, "binomi": 3, "poisson": [3, 4], "categor": 3, "multinomi": [3, 4], "connect": 3, "maximum": [3, 5], "likelihood": [3, 5], "estim": [3, 5], "exampl": [3, 4, 5], "mle": 3, "asymptot": 3, "normal": 3, "fisher": [3, 4], "inform": 3, "matrix": 3, "hypothesi": 3, "test": [3, 4], "wald": 3, "confid": [3, 4], "interv": [3, 4], "paramet": 3, "bayesian": [3, 4, 5], "credibl": 3, "demo": 3, "colleg": 3, "footbal": 3, "nation": 3, "championship": 3, "poll": 3, "result": 3, "compar": [3, 4], "actual": 3, "score": 3, "individu": 3, "drive": 3, "simul": 3, "come": 3, "full": 3, "circl": 3, "postgam": 3, "analysi": 3, "conclus": [3, 4, 5], "conting": [4, 5], "tabl": [4, 5], "motiv": 4, "question": [4, 5], "independ": 4, "sampl": 4, "hypergeometr": 4, "deriv": 4, "bay": 4, "rule": 4, "two": [4, 5], "proport": 4, "condit": 4, "simpson": 4, "paradox": 4, "log": [4, 5], "odd": 4, "ratio": 4, "delta": 4, "wai": [4, 5], "exact": 4, "setup": 5, "relationship": 5, "note": 5, "about": 5, "intercept": 5, "comput": 5, "convex": 5, "likelhood": 5, "converg": 5, "rate": 5, "bound": 5, "covari": 5, "patholog": 5, "separ": 5, "regim": 5, "regular": 5, "choos": 5, "hyperparamet": 5, "perspect": 5, "revisit": 5, "refer": 6}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})