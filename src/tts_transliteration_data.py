"""
Данные для транслитерации TTS: операторы Python и 2000 частых английских слов.
Модуль подключается из tts_client при предобработке текста для озвучивания.
"""

import re
import os
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# 1. ОПЕРАТОРЫ ЯЗЫКА PYTHON
# Символьные операторы: замена на русские эквиваленты для TTS.
# Порядок: от самых длинных к коротким (чтобы "==" обрабатывался раньше "=").
# ---------------------------------------------------------------------------
PYTHON_OPERATOR_SYMBOLS: Dict[str, str] = {
    # Составные операторы присваивания (самые длинные первыми)
    '**=': ' возведение в степень и присваивание ',
    '//=': ' целочисленное деление и присваивание ',
    '<<=': ' сдвиг влево и присваивание ',
    '>>=': ' сдвиг вправо и присваивание ',
    '^=': ' исключающее или и присваивание ',
    '|=': ' или и присваивание ',
    '&=': ' и и присваивание ',
    '%=': ' остаток и присваивание ',
    '@=': ' умножение матриц и присваивание ',
    '/=': ' деление и присваивание ',
    '*=': ' умножение и присваивание ',
    '-=': ' вычитание и присваивание ',
    '+=': ' сложение и присваивание ',
    # Двухсимвольные операторы
    '->': ' стрелка ',
    '...': ' многоточие ',
    '**': ' возведение в степень ',
    '//': ' целочисленное деление ',
    '<<': ' сдвиг влево ',
    '>>': ' сдвиг вправо ',
    '==': ' равно ',
    '!=': ' не равно ',
    '<=': ' меньше или равно ',
    '>=': ' больше или равно ',
    # Одиночные операторы
    '=': ' равно ',
    '<': ' меньше ',
    '>': ' больше ',
    '+': ' плюс ',
    # Одиночный '-' не заменяем здесь: в обычном тексте дефис (какой-то) не озвучивать как "минус".
    # "Минус" только между цифрами — обрабатывается в tts_client._preprocess_text().
    '*': ' умножить ',
    '/': ' разделить ',
    '%': ' остаток от деления ',
    '@': ' собака ',
    '&': ' амперсанд ',
    '|': ' вертикальная черта ',
    '^': ' крышка ',
    '~': ' тильда ',
}

# Ключевые слова Python (для озвучивания в коде/документации)
# Часть совпадает с частыми английскими словами; дубликаты в ENGLISH дают ту же транскрипцию.
PYTHON_KEYWORDS: Dict[str, str] = {
    'and': ' энд ', 'or': ' ор ', 'not': ' нот ', 'in': ' ин ', 'is': ' из ',
    'if': ' иф ', 'else': ' элс ', 'elif': ' элиф ', 'for': ' фор ', 'while': ' вайл ',
    'def': ' деф ', 'class': ' класс ', 'return': ' ретурн ', 'yield': ' йилд ',
    'pass': ' пас ', 'break': ' брейк ', 'continue': ' контину ',
    'raise': ' рейз ', 'try': ' трай ', 'except': ' эксепт ', 'finally': ' файнали ',
    'with': ' виз ', 'as': ' эз ', 'from': ' фром ', 'import': ' импорт ',
    'global': ' глоубл ', 'nonlocal': ' нонлоукал ', 'lambda': ' лямбда ',
    'assert': ' ассерт ', 'del': ' дел ', 'async': ' асинк ', 'await': ' авейт ',
    'match': ' мэтч ', 'case': ' кейс ',
    'True': ' тру ', 'False': ' фолс ', 'None': ' нон ',
}

# Сортированные по длине (убывание) для применения символов
_PYTHON_OP_SORTED = sorted(PYTHON_OPERATOR_SYMBOLS.keys(), key=len, reverse=True)


def _simple_english_to_russian(word: str) -> str:
    """
    Правило-based транслитерация английского слова в русскую транскрипцию.
    Используется для слов, для которых нет явной подстановки в EXPLICIT_ENGLISH.
    """
    if not word:
        return word
    w = word.lower()
    # Диграфы (обрабатываем до букв)
    w = re.sub(r'th', 'з', w)   # the->зе, think->зинк
    w = re.sub(r'sh', 'ш', w)
    w = re.sub(r'ch', 'ч', w)
    w = re.sub(r'ck', 'к', w)
    w = re.sub(r'ph', 'ф', w)
    w = re.sub(r'qu', 'кв', w)
    w = re.sub(r'ee', 'и', w)
    w = re.sub(r'oo', 'у', w)
    w = re.sub(r'ng', 'нг', w)
    # Одиночные буквы
    m = {
        'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'и', 'f': 'ф', 'g': 'г',
        'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
        'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'й', 'z': 'з',
    }
    res = []
    for c in w:
        res.append(m.get(c, c))
    return ''.join(res)


# ---------------------------------------------------------------------------
# 2. ЯВНЫЕ ТРАНСКРИПЦИИ для самых частых/сложных английских слов
# (остальные из 2000 получают _simple_english_to_russian)
# ---------------------------------------------------------------------------
EXPLICIT_ENGLISH: Dict[str, str] = {
    'the': 'зе', 'of': 'ов', 'and': 'энд', 'to': 'ту', 'a': 'эй', 'in': 'ин',
    'for': 'фор', 'is': 'из', 'on': 'он', 'that': 'зэт', 'by': 'бай', 'this': 'зис',
    'with': 'виз', 'i': 'ай', 'you': 'ю', 'it': 'ит', 'not': 'нот', 'or': 'ор',
    'be': 'би', 'are': 'ар', 'from': 'фром', 'at': 'эт', 'as': 'эз', 'your': 'ёр',
    'all': 'ол', 'have': 'хэв', 'new': 'ню', 'more': 'мор', 'an': 'эн', 'was': 'воз',
    'we': 'ви', 'will': 'вил', 'home': 'хоум', 'can': 'кэн', 'us': 'ас', 'about': 'эбаут',
    'if': 'иф', 'page': 'пейдж', 'my': 'май', 'has': 'хэз', 'search': 'сёрч', 'free': 'фри',
    'but': 'бат', 'our': 'аур', 'one': 'ван', 'other': 'азер', 'do': 'ду', 'no': 'ноу',
    'information': 'информейшн', 'time': 'тайм', 'they': 'зей', 'site': 'сайт', 'he': 'хи',
    'up': 'ап', 'may': 'мей', 'what': 'вот', 'which': 'вич', 'their': 'зэйр', 'news': 'нюз',
    'out': 'аут', 'use': 'юз', 'any': 'эни', 'there': 'зэар', 'see': 'си', 'only': 'онли',
    'so': 'соу', 'his': 'хиз', 'when': 'вен', 'contact': 'контакт', 'here': 'хиар',
    'business': 'бизнес', 'who': 'ху', 'web': 'веб', 'also': 'олсоу', 'now': 'нау',
    'help': 'хелп', 'get': 'гет', 'view': 'вью', 'online': 'онлайн', 'first': 'фёрст',
    'am': 'эм', 'been': 'бин', 'would': 'вуд', 'how': 'хау', 'were': 'вёр', 'me': 'ми',
    'some': 'сам', 'these': 'зиз', 'its': 'итс', 'like': 'лайк', 'service': 'сёрвис',
    'than': 'зэн', 'find': 'файнд', 'price': 'прайс', 'date': 'дэйт', 'back': 'бэк',
    'top': 'топ', 'people': 'пипл', 'had': 'хэд', 'list': 'лист', 'name': 'нам', 'just': 'джаст',
    'over': 'оувер', 'state': 'стэйт', 'year': 'йир', 'day': 'дэй', 'into': 'инту',
    'two': 'ту', 'health': 'хелс', 'world': 'ворлд', 'next': 'некст', 'used': 'юзд',
    'go': 'гоу', 'work': 'ворк', 'last': 'ласт', 'most': 'моуст', 'products': 'продактс',
    'music': 'мьюзик', 'buy': 'бай', 'data': 'дэйта', 'make': 'мейк', 'them': 'зэм',
    'should': 'шуд', 'product': 'продакт', 'system': 'систем', 'post': 'поуст',
    'her': 'хёр', 'city': 'сити', 'add': 'эд', 'policy': 'полиси', 'number': 'намбер',
    'such': 'сач', 'please': 'плиз', 'available': 'эвейлабл', 'copyright': 'копирайт',
    'support': 'саппорт', 'message': 'мессидж', 'after': 'афтер', 'best': 'бест',
    'software': 'софтвэр', 'then': 'зэн', 'good': 'гуд', 'video': 'видио', 'well': 'вел',
    'where': 'вэар', 'info': 'инфо', 'rights': 'райтс', 'public': 'паблик', 'books': 'букс',
    'high': 'хай', 'school': 'скул', 'through': 'сру', 'each': 'ич', 'links': 'линкс',
    'she': 'ши', 'review': 'ревью', 'years': 'йирс', 'order': 'ордер', 'very': 'вэри',
    'privacy': 'прайвеси', 'book': 'бук', 'items': 'айтемз', 'company': 'кампени',
    'read': 'рид', 'group': 'груп', 'need': 'нид', 'many': 'мэни', 'user': 'юзер',
    'said': 'сэд', 'does': 'даз', 'set': 'сет', 'under': 'андер', 'general': 'дженерал',
    'research': 'рисёрч', 'university': 'юнивёрсити', 'mail': 'мейл', 'full': 'фул',
    'map': 'мэп', 'program': 'программа', 'life': 'лайф', 'know': 'ноу', 'games': 'геймз',
    'way': 'вей', 'days': 'дэйз', 'management': 'мэнеджмент', 'part': 'парт', 'could': 'куд',
    'great': 'грэйт', 'united': 'юнайтид', 'hotel': 'хотел', 'real': 'рил', 'item': 'айтем',
    'international': 'интернэшнл', 'center': 'сэнтер', 'must': 'маст', 'store': 'стор',
    'travel': 'травл', 'comments': 'коментс', 'made': 'мейд', 'development': 'девелопмент',
    'report': 'рипорт', 'off': 'оф', 'member': 'мембер', 'details': 'дитейлз', 'line': 'лайн',
    'terms': 'тёрмз', 'before': 'бифор', 'did': 'дид', 'send': 'сенд', 'right': 'райт',
    'type': 'тайп', 'because': 'бикоз', 'local': 'лоукал', 'those': 'зоуз', 'using': 'юзинг',
    'results': 'ризалтс', 'office': 'офис', 'education': 'эдьюкейшн', 'national': 'нэшнл',
    'car': 'кар', 'design': 'дизайн', 'take': 'тейк', 'internet': 'интернет', 'address': 'эдрес',
    'community': 'комьюнити', 'within': 'визин', 'states': 'стэйтс', 'area': 'эриа',
    'want': 'вонт', 'phone': 'фон', 'shipping': 'шипинг', 'reserved': 'ризёрвд',
    'subject': 'сабджект', 'between': 'битуин', 'forum': 'форум', 'family': 'фэмили',
    'long': 'лонг', 'based': 'бэйст', 'code': 'код', 'show': 'шоу', 'even': 'ивен',
    'black': 'блэк', 'check': 'чек', 'special': 'спешл', 'prices': 'прайсиз',
    'website': 'вебсайт', 'index': 'индекс', 'being': 'биинг', 'women': 'вимен',
    'much': 'мач', 'sign': 'сайн', 'file': 'файл', 'link': 'линк', 'open': 'оупен',
    'today': 'тудэй', 'technology': 'текнолоджи', 'south': 'саут', 'case': 'кейс',
    'project': 'прожект', 'same': 'сейм', 'pages': 'пейджиз', 'version': 'вёршн',
    'section': 'секшн', 'own': 'оун', 'found': 'фаунд', 'sports': 'спортс', 'house': 'хаус',
    'related': 'рилейтид', 'security': 'сикьюрити', 'both': 'боуз', 'county': 'каунти',
    'american': 'американ', 'photo': 'фото', 'game': 'гейм', 'members': 'мемберс',
    'power': 'пауэр', 'while': 'вайл', 'care': 'кэар', 'network': 'нетворк', 'down': 'даун',
    'computer': 'компьютер', 'systems': 'системс', 'three': 'сри', 'total': 'тотал',
    'place': 'плейс', 'end': 'энд', 'following': 'фолоуинг', 'download': 'даунлоад',
    'him': 'хим', 'without': 'визаут', 'per': 'пёр', 'access': 'аксес', 'think': 'синк',
    'north': 'норс', 'resources': 'ризорсиз', 'current': 'карент', 'posts': 'поустс',
    'big': 'биг', 'media': 'мидиа', 'law': 'ло', 'control': 'контрол', 'water': 'вотер',
    'history': 'хистори', 'pictures': 'пикчерс', 'size': 'сайз', 'art': 'арт',
    'personal': 'персонл', 'since': 'синс', 'including': 'инклудинг', 'guide': 'гайд',
    'shop': 'шоп', 'directory': 'дайректори', 'board': 'борд', 'location': 'лоукейшн',
    'change': 'чейндж', 'white': 'вайт', 'text': 'текст', 'small': 'смол', 'rating': 'рейтинг',
    'rate': 'рейт', 'government': 'гавернмент', 'children': 'чилдрен', 'during': 'дьюринг',
    'usa': 'юэсэй', 'return': 'ритёрн', 'students': 'стьюдентс', 'shopping': 'шоппинг',
    'account': 'акаунт', 'times': 'таймс', 'sites': 'сайтс', 'level': 'левл',
    'digital': 'диджитал', 'profile': 'профайл', 'previous': 'привиас', 'form': 'форм',
    'events': 'ивентс', 'love': 'лав', 'old': 'оулд', 'john': 'джон', 'main': 'мейн',
    'call': 'кол', 'hours': 'аурс', 'image': 'имидж', 'department': 'дипартмент',
    'title': 'тайтл', 'description': 'дискрипшн', 'insurance': 'иншуранс', 'another': 'аназер',
    'why': 'вай', 'shall': 'шэл', 'property': 'проперти', 'class': 'класс', 'still': 'стил',
    'money': 'мани', 'quality': 'кволити', 'every': 'эври', 'listing': 'листинг',
    'content': 'контент', 'country': 'кантри', 'private': 'прайвет', 'little': 'литл',
    'visit': 'визит', 'save': 'сейв', 'tools': 'тулс', 'low': 'лоу', 'reply': 'риплай',
    'customer': 'кастомер', 'compare': 'компэар', 'movies': 'мувиз', 'include': 'инклуд',
    'college': 'колидж', 'value': 'вэлю', 'article': 'артикл', 'man': 'мэн', 'card': 'кард',
    'jobs': 'джобс', 'provide': 'провайд', 'food': 'фуд', 'source': 'сорс', 'author': 'отар',
    'different': 'дифрент', 'press': 'прес', 'learn': 'лёрн', 'sale': 'сейл', 'around': 'эраунд',
    'print': 'принт', 'course': 'корс', 'job': 'джоб', 'canada': 'канада', 'process': 'просес',
    'teen': 'тин', 'room': 'рум', 'stock': 'сток', 'training': 'трэйнинг', 'too': 'ту',
    'credit': 'кредит', 'point': 'поинт', 'join': 'джойн', 'science': 'сайенс', 'men': 'мен',
    'advanced': 'адванст', 'west': 'вест', 'sales': 'сейлз', 'look': 'лук', 'english': 'инглиш',
    'left': 'левт', 'team': 'тим', 'estate': 'истейт', 'box': 'бокс', 'conditions': 'кондишнс',
    'select': 'силект', 'windows': 'виндоуз', 'photos': 'фотоуз', 'thread': 'сред',
    'week': 'вик', 'category': 'кэтигори', 'note': 'ноут', 'live': 'лив', 'large': 'лардж',
    'gallery': 'гэлери', 'table': 'тейбл', 'register': 'реджистер', 'however': 'хауэвер',
    'june': 'джун', 'october': 'октоубер', 'november': 'ноувембер', 'market': 'маркет',
    'library': 'лайбрери', 'really': 'риели', 'action': 'экшн', 'start': 'старт',
    'series': 'сириз', 'model': 'модел', 'features': 'фичерс', 'air': 'эар', 'industry': 'индастри',
    'plan': 'план', 'human': 'хьюман', 'provided': 'провайдед', 'tv': 'тиви', 'yes': 'йес',
    'required': 'рикуайерд', 'second': 'секонд', 'hot': 'хот', 'cost': 'кост', 'movie': 'муви',
    'march': 'марч', 'september': 'септембер', 'better': 'беттер', 'say': 'сей', 'questions': 'квестшнс',
    'july': 'джулай', 'going': 'гоуинг', 'medical': 'медикл', 'test': 'тест', 'friend': 'френд',
    'come': 'кам', 'server': 'сёрвер', 'study': 'стади', 'application': 'эпликейшн', 'cart': 'карт',
    'staff': 'стаф', 'articles': 'артиклз', 'feedback': 'фидбэк', 'again': 'эгейн', 'play': 'плей',
    'looking': 'лукинг', 'issues': 'ишьюз', 'april': 'эйприл', 'never': 'невер', 'users': 'юзерс',
    'complete': 'комплит', 'street': 'стрит', 'topic': 'топик', 'comment': 'комент',
    'financial': 'файнэншл', 'things': 'синз', 'working': 'воркинг', 'against': 'эгейнст',
    'standard': 'стэндерд', 'tax': 'тэкс', 'person': 'пёрсон', 'below': 'билоу', 'mobile': 'мобайл',
    'less': 'лес', 'got': 'гот', 'blog': 'блог', 'party': 'парти', 'payment': 'пеймент',
    'equipment': 'иквипмент', 'login': 'логин', 'student': 'стьюдент', 'let': 'лет',
    'programs': 'программаз', 'offers': 'офферс', 'legal': 'лигл', 'above': 'эбав',
    'recent': 'рисент', 'park': 'парк', 'stores': 'сторс', 'side': 'сайд', 'act': 'экт',
    'problem': 'проблем', 'red': 'ред', 'give': 'гив', 'memory': 'мемори', 'performance': 'пёрформенс',
    'social': 'соушл', 'august': 'огэст', 'quote': 'квоут', 'language': 'лэнгвидж',
    'story': 'стори', 'sell': 'сел', 'options': 'опшнс', 'experience': 'икспириенс',
    'rates': 'рейтс', 'create': 'криэйт', 'key': 'ки', 'body': 'боди', 'young': 'янг',
    'america': 'америка', 'important': 'импортант', 'field': 'филд', 'few': 'фью',
    'east': 'ист', 'paper': 'пейпер', 'single': 'сингл', 'age': 'ейдж', 'activities': 'эктивйтиз',
    'club': 'клаб', 'example': 'игзампл', 'girls': 'гёрлз', 'additional': 'эдишнл',
    'password': 'пассворд', 'latest': 'лейтест', 'something': 'самсинг', 'road': 'роуд',
    'gift': 'гифт', 'question': 'квестшн', 'changes': 'чейнджиз', 'night': 'найт',
    'hard': 'хард', 'pay': 'пей', 'four': 'фор', 'status': 'стэйтус', 'browse': 'брауз',
    'issue': 'ишью', 'range': 'рейндж', 'building': 'билдинг', 'seller': 'селлер',
    'court': 'корт', 'february': 'фебруари', 'always': 'олвейз', 'result': 'ризалт',
    'audio': 'одио', 'light': 'лайт', 'write': 'райт', 'war': 'вор', 'offer': 'оффер',
    'blue': 'блу', 'groups': 'групс', 'easy': 'изи', 'given': 'гивен', 'files': 'файлз',
    'event': 'ивент', 'release': 'рилис', 'analysis': 'энэлисис', 'request': 'риквест',
    'china': 'чайна', 'making': 'мейкинг', 'picture': 'пикчер', 'needs': 'нидз',
    'possible': 'посибл', 'might': 'майт', 'professional': 'профешнл', 'yet': 'ет',
    'month': 'манс', 'major': 'мейджор', 'star': 'стар', 'areas': 'эриаз', 'future': 'фьючер',
    'space': 'спейс', 'committee': 'комити', 'hand': 'хэнд', 'sun': 'сан', 'cards': 'кардс',
    'problems': 'проблемс', 'london': 'лондон', 'meeting': 'митинг', 'become': 'бикам',
    'interest': 'интерест', 'child': 'чайлд', 'keep': 'кип', 'enter': 'энтер',
    'california': 'калифорниа', 'share': 'шэар', 'similar': 'симилар', 'garden': 'гарден',
    'schools': 'скулз', 'million': 'миллион', 'added': 'эдед', 'reference': 'референс',
    'companies': 'кампениз', 'listed': 'листид', 'baby': 'бейби', 'learning': 'лёрнинг',
    'energy': 'энерджи', 'run': 'ран', 'delivery': 'диливери', 'net': 'нет',
    'popular': 'популар', 'term': 'тёрм', 'film': 'филм', 'stories': 'сториз',
    'put': 'пут', 'computers': 'компьютерс', 'journal': 'джёрнл', 'reports': 'рипортс',
    'try': 'трай', 'welcome': 'велком', 'central': 'сентрал', 'images': 'имиджиз',
    'president': 'президент', 'notice': 'нотис', 'god': 'год', 'original': 'ориджинл',
    'head': 'хед', 'radio': 'рейдио', 'until': 'антинл', 'cell': 'сел', 'color': 'калер',
    'self': 'селф', 'council': 'каунсил', 'away': 'эвей', 'includes': 'инклудз',
    'track': 'трэк', 'australia': 'острейлиа', 'discussion': 'дискашн', 'archive': 'архайв',
    'once': 'ванс', 'others': 'азерс', 'entertainment': 'энтертеймент', 'agreement': 'эгримент',
    'format': 'формат', 'least': 'лист', 'society': 'сошайети', 'months': 'манс',
    'safety': 'сейфти', 'friends': 'френдз', 'sure': 'шур', 'trade': 'трейд',
    'edition': 'идишн', 'cars': 'карс', 'messages': 'мессиджиз', 'marketing': 'маркетинг',
    'tell': 'тел', 'further': 'фёрзер', 'updated': 'апдейтид', 'association': 'асоушиейшн',
    'able': 'эйбл', 'having': 'хэвинг', 'provides': 'провайдз', 'fun': 'фан',
    'already': 'олреди', 'green': 'грин', 'studies': 'стадиз', 'close': 'клоуз',
    'common': 'комон', 'drive': 'драйв', 'specific': 'списифик', 'several': 'сиврел',
    'gold': 'гоулд', 'living': 'ливинг', 'collection': 'колекшн', 'called': 'колд',
    'short': 'шорт', 'arts': 'артс', 'lot': 'лот', 'ask': 'аск', 'display': 'дисплей',
    'limited': 'лимитед', 'solutions': 'солюшнс', 'means': 'минз', 'director': 'дайректор',
    'daily': 'дэйли', 'beach': 'бич', 'past': 'паст', 'natural': 'нэчурал', 'whether': 'ветер',
    'due': 'дью', 'electronics': 'илектроникс', 'five': 'файв', 'upon': 'апон',
    'period': 'пириод', 'planning': 'плэнинг', 'database': 'дэйтабэйс', 'says': 'сейз',
    'official': 'офишл', 'weather': 'ветер', 'land': 'лэнд', 'average': 'эверидж',
    'done': 'дан', 'technical': 'текникл', 'window': 'виндоу', 'france': 'франс',
    'region': 'риджн', 'island': 'айленд', 'record': 'рекорд', 'direct': 'дайрект',
    'microsoft': 'майкрософт', 'conference': 'конференс', 'environment': 'инвайронмент',
    'records': 'рекордс', 'district': 'дистрикт', 'calendar': 'календар', 'costs': 'костс',
    'style': 'стайл', 'front': 'франт', 'statement': 'стэйтмент', 'update': 'апдейт',
    'parts': 'партс', 'ever': 'эвер', 'downloads': 'даунлоадс', 'early': 'ёрли',
    'miles': 'майлс', 'sound': 'саунд', 'resource': 'ризорс', 'present': 'презент',
    'applications': 'эпликейшнс', 'either': 'айзер', 'ago': 'эгоу', 'document': 'документ',
    'word': 'ворд', 'works': 'воркс', 'material': 'матэриал', 'bill': 'бил', 'written': 'ритн',
    'talk': 'ток', 'federal': 'федерал', 'hosting': 'хоустинг', 'rules': 'рулс',
    'final': 'файнал', 'adult': 'адалт', 'tickets': 'тикетс', 'thing': 'синг',
    'centre': 'сэнтер', 'requirements': 'риквайрментс', 'via': 'вайа', 'cheap': 'чип',
    'nude': 'нюд', 'kids': 'кидс', 'finance': 'файненс', 'true': 'тру', 'minutes': 'минитс',
    'else': 'элс', 'mark': 'марк', 'third': 'сёрд', 'rock': 'рок', 'gifts': 'гифтс',
    'europe': 'юроп', 'reading': 'ридинг', 'topics': 'топикс', 'bad': 'бэд',
    'individual': 'индивижуал', 'tips': 'типс', 'plus': 'плас', 'auto': 'отоу',
    'cover': 'кавер', 'usually': 'южуали', 'edit': 'эдит', 'together': 'тугезер',
    'videos': 'видиоз', 'percent': 'пёрсент', 'fast': 'фаст', 'function': 'фанкшн',
    'fact': 'фэкт', 'unit': 'юнит', 'getting': 'геттинг', 'global': 'глоубл', 'tech': 'тек',
    'meet': 'мит', 'far': 'фар', 'economic': 'икономик', 'player': 'плейер',
    'projects': 'прожектс', 'lyrics': 'лирикс', 'often': 'офн', 'subscribe': 'сабскрайб',
    'submit': 'сабмит', 'germany': 'джёрмани', 'amount': 'амаунт', 'watch': 'вотч',
    'included': 'инклудед', 'feel': 'фил', 'though': 'зоу', 'bank': 'бэнк', 'risk': 'риск',
    'thanks': 'сэнкс', 'everything': 'эврисинг', 'deals': 'дилс', 'various': 'вэриас',
    'words': 'вордс', 'linux': 'линукс', 'production': 'продакшн', 'commercial': 'комёршл',
    'james': 'джеймс', 'weight': 'вейт', 'town': 'таун', 'heart': 'харт',
    'advertising': 'эдвертайзинг', 'received': 'рисивд', 'choose': 'чуз', 'treatment': 'тритмент',
    'newsletter': 'нюзлеттер', 'archives': 'архайвз', 'points': 'поинтс', 'knowledge': 'нолидж',
    'magazine': 'мэгэзин', 'error': 'эрор', 'camera': 'кэмера', 'girl': 'гёрл',
    'currently': 'карентли', 'construction': 'констракшн', 'toys': 'тойз',
    'registered': 'реджистерд', 'clear': 'клир', 'golf': 'голф', 'receive': 'рисив',
    'domain': 'домейн', 'methods': 'мэтодс', 'chapter': 'чэптер', 'makes': 'мейкс',
    'protection': 'протекшн', 'policies': 'полисиз', 'loan': 'лоун', 'wide': 'вайд',
    'beauty': 'бьюти', 'manager': 'мэнеджер', 'india': 'индиа', 'position': 'позишн',
    'taken': 'тейкн', 'sort': 'сорт', 'listings': 'листингс', 'models': 'моделз',
    'michael': 'майкл', 'known': 'ноун', 'half': 'хаф', 'cases': 'кейсиз', 'step': 'степ',
    'engineering': 'инджиниринг', 'florida': 'флорида', 'simple': 'симпл', 'quick': 'квик',
    'none': 'нан', 'wireless': 'вайрлес', 'license': 'лайсенс', 'paul': 'пол',
    'friday': 'фрайдей', 'lake': 'лейк', 'whole': 'хоул', 'annual': 'эньюал',
    'published': 'паблишт', 'later': 'лейтер', 'basic': 'бейсик', 'sony': 'сони',
    'shows': 'шоуз', 'corporate': 'корпорет', 'google': 'гугл', 'church': 'чёрч',
    'method': 'мэтод', 'purchase': 'пёрчас', 'customers': 'кастомерс', 'active': 'актив',
    'response': 'риспонс', 'practice': 'прэктис', 'hardware': 'хардвэр', 'figure': 'фигур',
    'materials': 'матэриалз', 'fire': 'файр', 'holiday': 'холидей', 'chat': 'чэт',
    'enough': 'инаф', 'designed': 'дизайнд', 'along': 'элонг', 'among': 'эманг',
    'death': 'дес', 'writing': 'райтинг', 'speed': 'спид', 'html': 'эйчтиэмэль',
    'countries': 'кантриз', 'loss': 'лос', 'face': 'фейс', 'brand': 'брэнд',
    'discount': 'дискаунт', 'higher': 'хайер', 'effects': 'ифэктс', 'created': 'криейтид',
    'remember': 'римембер', 'standards': 'стэндердз', 'oil': 'ойл', 'bit': 'бит',
    'yellow': 'йеллоу', 'political': 'политикл', 'increase': 'инкрис', 'advertise': 'эдвертайз',
    'kingdom': 'кингдом', 'base': 'бейс', 'near': 'нир', 'environmental': 'инвайронментл',
    'thought': 'сот', 'stuff': 'стаф', 'french': 'френч', 'storage': 'сторидж',
    'japan': 'джэпэн', 'doing': 'дуинг', 'loans': 'лоунз', 'shoes': 'шуз', 'entry': 'энтри',
    'stay': 'стей', 'nature': 'нэйчер', 'orders': 'ордерз', 'availability': 'эвейлабилити',
    'africa': 'африка', 'summary': 'самери', 'turn': 'тёрн', 'mean': 'мин', 'growth': 'гроут',
    'notes': 'ноутс', 'agency': 'эйдженси', 'king': 'кинг', 'monday': 'мандей',
    'european': 'юропиан', 'activity': 'эктивети', 'copy': 'копи', 'although': 'олзоу',
    'drug': 'драг', 'pics': 'пикс', 'western': 'вестерн', 'income': 'инкам', 'force': 'форс',
    'cash': 'кэш', 'employment': 'имплоймент', 'overall': 'оуверол', 'bay': 'бей',
    'river': 'ривер', 'commission': 'комишн', 'package': 'пэкейдж', 'contents': 'контентс',
    'seen': 'син', 'players': 'плейерз', 'engine': 'инджин', 'port': 'порт',
    'album': 'элбум', 'regional': 'риджнл', 'stop': 'стоп', 'supplies': 'саплайз',
    'started': 'стартид', 'administration': 'администрэйшн', 'institute': 'инститют',
    'views': 'вьюз', 'plans': 'планз', 'double': 'дабл', 'dog': 'дог', 'build': 'билд',
    'screen': 'скрин', 'exchange': 'иксчейндж', 'types': 'тайпс', 'soon': 'сун',
    'sponsored': 'спонсорд', 'lines': 'лайнз', 'electronic': 'илектроник',
    'continue': 'контину', 'across': 'экросс', 'benefits': 'бенефитс', 'needed': 'нидед',
    'season': 'сизн', 'apply': 'эплай', 'someone': 'самван', 'held': 'хелд',
    'anything': 'энисинг', 'printer': 'принтер', 'condition': 'кондишн', 'effective': 'ифэктив',
    'believe': 'билив', 'organization': 'органайзейшн', 'effect': 'ифэкт', 'asked': 'аскт',
    'mind': 'майнд', 'sunday': 'сандей', 'selection': 'силекшн', 'casino': 'казино',
    'pdf': 'пидиэф', 'lost': 'лост', 'tour': 'тур', 'menu': 'мёню', 'volume': 'воляюм',
    'cross': 'кросс', 'anyone': 'эниван', 'mortgage': 'моргидж', 'hope': 'хоуп',
    'silver': 'силвер', 'corporation': 'корпорейшн', 'wish': 'виш', 'inside': 'инсайд',
    'solution': 'солюшн', 'mature': 'матюр', 'role': 'роул', 'rather': 'разер',
    'weeks': 'викс', 'addition': 'эдишн', 'came': 'кейм', 'supply': 'саплай',
    'nothing': 'насинг', 'certain': 'сёртан', 'executive': 'игзекьютив', 'running': 'раниннг',
    'lower': 'лоуер', 'necessary': 'несесери', 'union': 'юнион', 'according': 'экординг',
    'clothing': 'клоузинг', 'particular': 'партикулар', 'fine': 'файн', 'names': 'неймз',
    'robert': 'роберт', 'homepage': 'хоумпейдж', 'hour': 'аур', 'gas': 'гэс',
    'skills': 'скилз', 'six': 'сикс', 'bush': 'буш', 'islands': 'айлендс',
    'advice': 'эдвайс', 'career': 'карир', 'military': 'милитери', 'rental': 'рентал',
    'decision': 'дисижн', 'leave': 'лив', 'british': 'бритиш', 'teens': 'тинз',
    'huge': 'хьюдж', 'woman': 'вуман', 'facilities': 'фэсилитиз', 'zip': 'зип',
    'bid': 'бид', 'kind': 'кайнд', 'sellers': 'селлерз', 'middle': 'мидл', 'move': 'мув',
    'cable': 'кейбл', 'opportunities': 'опортьюнитиз', 'taking': 'тейкинг', 'values': 'вэлюз',
    'division': 'дивижн', 'coming': 'каминг', 'tuesday': 'тьюздей', 'object': 'обджект',
    'appropriate': 'эпроуприет', 'machine': 'машин', 'logo': 'логоу', 'length': 'ленгс',
    'actually': 'экчуали', 'nice': 'найс', 'score': 'скор', 'statistics': 'стэтистикс',
    'client': 'клайент', 'returns': 'ритёрнз', 'capital': 'кэпител', 'follow': 'фоллоу',
    'sample': 'сампл', 'investment': 'инвестмент', 'sent': 'сент', 'shown': 'шоун',
    'saturday': 'сэтердей', 'christmas': 'кристмас', 'england': 'ингленд', 'culture': 'калчер',
    'band': 'бэнд', 'flash': 'флэш', 'lead': 'лид', 'george': 'джордж', 'choice': 'чойс',
    'went': 'вент', 'starting': 'стартинг', 'registration': 'реджистрейшн',
    'thursday': 'сёрздей', 'courses': 'корсиз', 'consumer': 'консюмер', 'airport': 'эрпорт',
    'foreign': 'форейн', 'artist': 'артист', 'outside': 'аутсайд', 'furniture': 'фёрничер',
    'levels': 'левлз', 'channel': 'чэнл', 'letter': 'леттер', 'mode': 'моуд',
    'phones': 'фонз', 'ideas': 'айдиаз', 'wednesday': 'веднесдей', 'structure': 'стракчер',
    'fund': 'фанд', 'summer': 'самер', 'allow': 'элау', 'degree': 'дигри', 'contract': 'контракт',
    'button': 'батн', 'releases': 'рилисиз', 'homes': 'хоумз', 'super': 'супер',
    'male': 'мейл', 'matter': 'мэтер', 'custom': 'кастом', 'virginia': 'верджиниа',
    'almost': 'олмоуст', 'took': 'тук', 'located': 'лоукейтид', 'multiple': 'малтипл',
    'asian': 'эйшн', 'distribution': 'дистрибьюшн', 'editor': 'эдитор', 'industrial': 'индастриал',
    'cause': 'коз', 'potential': 'потаншл', 'song': 'сонг', 'focus': 'фоукус', 'late': 'лейт',
    'fall': 'фол', 'featured': 'фичерд', 'idea': 'айдиа', 'rooms': 'румз', 'female': 'фимейл',
    'responsible': 'риспонсибл', 'communications': 'комьюникейшнс', 'win': 'вин',
    'associated': 'асоушиейтид', 'thomas': 'томас', 'primary': 'праймери', 'cancer': 'кэнсер',
    'numbers': 'намберс', 'reason': 'ризн', 'tool': 'тул', 'browser': 'браузер',
    'spring': 'спринг', 'foundation': 'фаундейшн', 'answer': 'ансер', 'voice': 'войс',
    'friendly': 'френдли', 'schedule': 'скедул', 'documents': 'документс',
    'communication': 'комьюникейшн', 'purpose': 'пёрпас', 'feature': 'фичер', 'bed': 'бед',
    'comes': 'камз', 'police': 'полис', 'everyone': 'эвриван', 'independent': 'индипендент',
    'approach': 'эпроуч', 'cameras': 'кэмераз', 'brown': 'браун', 'physical': 'физикл',
    'operating': 'оперейтинг', 'hill': 'хил', 'maps': 'мэпс', 'medicine': 'медисин',
    'deal': 'дил', 'hold': 'хоулд', 'ratings': 'рейтингз', 'chicago': 'чикаго',
    'forms': 'формз', 'glass': 'глас', 'happy': 'хэпи', 'smith': 'смит', 'wanted': 'вонтид',
    'developed': 'дивелопд', 'thank': 'сэнк', 'safe': 'сейф', 'unique': 'юник',
    'survey': 'сёрвей', 'prior': 'прайор', 'telephone': 'телефон', 'sport': 'спорт',
    'ready': 'реди', 'feed': 'фид', 'animal': 'энимал', 'sources': 'сорсиз',
    'mexico': 'мексико', 'population': 'популейшн', 'regular': 'регулар', 'secure': 'сикьюр',
    'navigation': 'нэвигейшн', 'operations': 'оперейшнс', 'therefore': 'зэарфор',
    'simply': 'симпли', 'evidence': 'эвиденс', 'station': 'стэйшн', 'christian': 'кристиан',
    'round': 'раунд', 'paypal': 'пейпал', 'favorite': 'фейворит', 'understand': 'андерстэнд',
    'option': 'опшн', 'master': 'мастер', 'valley': 'вэли', 'recently': 'рисентли',
    'probably': 'проббли', 'rentals': 'ренталз', 'sea': 'си', 'built': 'билт',
    'publications': 'пабликейшнс', 'blood': 'блад', 'cut': 'кат', 'worldwide': 'ворлдвайд',
    'improve': 'импрув', 'connection': 'коннекшн', 'publisher': 'паблишер', 'hall': 'хол',
    'larger': 'ларджер', 'networks': 'нетворкс', 'earth': 'ёрс', 'parents': 'пэарентс',
    'nokia': 'нокиа', 'impact': 'импэкт', 'transfer': 'трансфёр', 'introduction': 'интродакшн',
    'kitchen': 'кичен', 'strong': 'стронг', 'tel': 'тел', 'carolina': 'каролайна',
    'wedding': 'веддинг', 'properties': 'пропертиз', 'hospital': 'хоспител',
    'ground': 'граунд', 'overview': 'оувервью', 'ship': 'шип', 'accommodation': 'экомодейшн',
    'owners': 'оунерз', 'disease': 'дизиз', 'excellent': 'экселент', 'paid': 'пейд',
    'italy': 'итали', 'perfect': 'пёрфект', 'hair': 'хэар', 'opportunity': 'опортьюнити',
    'kit': 'кит', 'classic': 'класик', 'basis': 'бейсис', 'command': 'команд',
    'cities': 'ситиз', 'william': 'вильям', 'express': 'икспрес', 'award': 'эворд',
    'distance': 'дистанс', 'tree': 'три', 'peter': 'питер', 'assessment': 'ассесмент',
    'ensure': 'иншур', 'thus': 'зас', 'wall': 'вол', 'involved': 'инволвд',
    'extra': 'экстра', 'especially': 'испешали', 'interface': 'интерфейс', 'partners': 'партнерз',
    'budget': 'баджет', 'rated': 'рейтид', 'guides': 'гайдз', 'success': 'саксес',
    'maximum': 'мэксимум', 'operation': 'оперейшн', 'existing': 'игзистинг', 'quite': 'квайт',
    'selected': 'силектид', 'boy': 'бой', 'amazon': 'амазон', 'patients': 'пейшенс',
    'restaurants': 'ресторантс', 'beautiful': 'бьютифул', 'warning': 'ворнинг', 'wine': 'вайн',
    'locations': 'лоукейшнс', 'horse': 'хорс', 'vote': 'воут', 'forward': 'форвард',
    'flowers': 'флауэрз', 'stars': 'старз', 'significant': 'сигнификент', 'lists': 'листс',
    'technologies': 'текнолоджиз', 'owner': 'оунер', 'retail': 'ритейл', 'animals': 'энималз',
    'useful': 'юсфул', 'directly': 'дайректли', 'manufacturer': 'мэньюфэкчерер',
    'ways': 'вейз', 'son': 'сан', 'providing': 'провайдинг', 'rule': 'рул', 'mac': 'мэк',
    'housing': 'хаузинг', 'takes': 'тейкс', 'bring': 'бринг', 'catalog': 'кэтэлог',
    'searches': 'сёрчиз', 'trying': 'трайинг', 'mother': 'мазер', 'authority': 'оторити',
    'considered': 'консидерд', 'told': 'тоулд', 'traffic': 'трэфик', 'programme': 'программа',
    'joined': 'джойнд', 'input': 'инпут', 'strategy': 'стрэтиджи', 'feet': 'фит',
    'agent': 'эйджент', 'valid': 'вэлид', 'modern': 'модерн', 'senior': 'синиор',
    'ireland': 'айрленд', 'sexy': 'секси', 'teaching': 'тичинг', 'door': 'дор',
    'grand': 'грэнд', 'testing': 'тестинг', 'trial': 'трайал', 'charge': 'чардж',
    'units': 'юнитс', 'instead': 'инстэд', 'canadian': 'кэйнидиан', 'cool': 'кул',
    'normal': 'нормал', 'wrote': 'роут', 'enterprise': 'энтерпрайз', 'ships': 'шипс',
    'entire': 'интайр', 'educational': 'эдьюкейшнл', 'leading': 'лидинг', 'metal': 'метал',
    'positive': 'позитив', 'fitness': 'фитнес', 'chinese': 'чайниз', 'opinion': 'опинион',
    'asia': 'эйша', 'football': 'футбол', 'abstract': 'эбстрэкт', 'uses': 'юзиз',
    'output': 'аутпут', 'funds': 'фандз', 'greater': 'грэйтер', 'likely': 'лайкли',
    'develop': 'дивелоп', 'employees': 'имплойиз', 'artists': 'артистс',
    'alternative': 'олтёрнатив', 'processing': 'просесинг', 'responsibility': 'риспонсибилити',
    'resolution': 'резолюшн', 'java': 'джава', 'guest': 'гест', 'seems': 'симз',
    'publication': 'пабликейшн', 'pass': 'пас', 'relations': 'рилейшнс', 'trust': 'траст',
    'van': 'вэн', 'contains': 'контейнз', 'session': 'сешн', 'photography': 'фотография',
    'republic': 'рипаблик', 'fees': 'физ', 'components': 'компоунентс', 'vacation': 'вейкейшн',
    'century': 'сэнчури', 'academic': 'экадемик', 'assistance': 'асистенс',
    'completed': 'комплитид', 'skin': 'скин', 'graphics': 'графикс', 'indian': 'индиан',
    'expected': 'икспектид', 'ring': 'ринг', 'grade': 'грэйд', 'dating': 'дэйтинг',
    'pacific': 'пасифик', 'mountain': 'маунтин', 'organizations': 'органайзейшнс',
    'pop': 'поп', 'filter': 'филтер', 'mailing': 'мейлинг', 'vehicle': 'виакл',
    'longer': 'лонгер', 'consider': 'консидер', 'northern': 'норзерн', 'behind': 'бихайнд',
    'panel': 'пэнл', 'floor': 'флор', 'german': 'джёрман', 'buying': 'байинг',
    'match': 'мэтч', 'proposed': 'пропоузд', 'default': 'дифолт', 'require': 'риквайр',
    'iraq': 'ирак', 'boys': 'бойз', 'outdoor': 'аутдор', 'deep': 'дип', 'morning': 'морнинг',
    'otherwise': 'азервайз', 'allows': 'элауз', 'rest': 'рест', 'protein': 'проутин',
    'plant': 'плант', 'reported': 'рипортид', 'hit': 'хит', 'transportation': 'транспортейшн',
    'pool': 'пул', 'mini': 'мини', 'politics': 'политикс', 'partner': 'партнер',
    'disclaimer': 'дисклеймер', 'authors': 'отарз', 'boards': 'бордс', 'faculty': 'фэкулти',
    'parties': 'партиз', 'fish': 'фиш', 'membership': 'мембершип', 'mission': 'мишн',
    'eye': 'ай', 'string': 'стринг', 'sense': 'сенс', 'modified': 'модифайд',
    'pack': 'пэк', 'released': 'рилист', 'stage': 'стейдж', 'internal': 'интернал',
    'goods': 'гудз', 'recommended': 'рекомендед', 'born': 'борн', 'unless': 'анлес',
    'richard': 'ричард', 'detailed': 'дитейлд', 'japanese': 'джэпаниз', 'race': 'рейс',
    'approved': 'эпрувд', 'background': 'бэкграунд', 'target': 'таргет', 'except': 'иксепт',
    'character': 'кэрэктер', 'maintenance': 'мейнтенанс', 'ability': 'эбилити',
    'maybe': 'мейби', 'functions': 'фанкшнс', 'moving': 'мувинг', 'brands': 'брэндз',
    'places': 'плейсиз', 'php': 'пиэйчпи', 'pretty': 'прити', 'trademarks': 'трэйдмаркс',
    'spain': 'спейн', 'southern': 'сазерн', 'yourself': 'ёрселф', 'etc': 'этсетера',
    'winter': 'винтер', 'battery': 'бэтери', 'youth': 'юс', 'pressure': 'прешер',
    'submitted': 'сабмитид', 'boston': 'бостон', 'debt': 'дет', 'keywords': 'кивордс',
    'medium': 'мидиам', 'television': 'телевижн', 'interested': 'интерестид', 'core': 'кор',
    'break': 'брейк', 'purposes': 'пёрпосиз', 'throughout': 'сруаут', 'sets': 'сетс',
    'dance': 'данс', 'wood': 'вуд', 'itself': 'итселф', 'defined': 'дифайнд',
    'papers': 'пейперз', 'playing': 'плейинг', 'fee': 'фи', 'studio': 'стюдио',
    'reader': 'ридер', 'virtual': 'вёрчуал', 'device': 'дивайс', 'established': 'истэблишт',
    'answers': 'ансерз', 'rent': 'рент', 'remote': 'римоут', 'dark': 'дарк',
    'programming': 'программинг', 'external': 'икстернал', 'apple': 'эпл',
    'regarding': 'ригардинг', 'instructions': 'инстракшнс', 'offered': 'офферд',
    'theory': 'сиори', 'enjoy': 'инджой', 'remove': 'римув', 'aid': 'ейд',
    'surface': 'сёрфейс', 'minimum': 'минимум', 'visual': 'вижуал', 'host': 'хоуст',
    'variety': 'вэрайети', 'teachers': 'тичерз', 'manual': 'мэнюал', 'block': 'блок',
    'subjects': 'сабджектс', 'agents': 'эйджентс', 'increased': 'инкрист', 'repair': 'рипэар',
    'fair': 'фэар', 'civil': 'сивил', 'steel': 'стил', 'understanding': 'андерстэндинг',
    'songs': 'сонгз', 'fixed': 'фикст', 'wrong': 'ронг', 'beginning': 'бигининг',
    'hands': 'хэндз', 'associates': 'асоушиейтс', 'finally': 'файнали', 'updates': 'апдейтс',
    'desktop': 'десктоп', 'classes': 'класиз', 'gets': 'гетс', 'sector': 'сектор',
    'capacity': 'капэсити', 'requires': 'риквайрс', 'jersey': 'джёрси', 'fat': 'фэт',
    'fully': 'фулли', 'father': 'фазер', 'electric': 'илектрик', 'instruments': 'инструментс',
    'quotes': 'квоутс', 'officer': 'офисер', 'driver': 'драйвер', 'businesses': 'бизнесиз',
    'dead': 'дед', 'respect': 'риспект', 'unknown': 'анноун', 'specified': 'спесифайд',
    'restaurant': 'ресторант', 'trip': 'трип', 'worth': 'ворс', 'procedures': 'просиджерз',
    'poor': 'пур', 'teacher': 'тичер', 'eyes': 'айз', 'relationship': 'рилейшншип',
    'workers': 'воркерс', 'farm': 'фарм', 'georgia': 'джорджиа', 'peace': 'пис',
    'traditional': 'традишнл', 'campus': 'кэмпас', 'showing': 'шоуинг', 'creative': 'криейтив',
    'coast': 'коуст', 'benefit': 'бенефит', 'progress': 'прогрес', 'funding': 'фандинг',
    'devices': 'дивайсиз', 'lord': 'лорд', 'grant': 'грант', 'agree': 'эгри',
    'fiction': 'фикшн', 'hear': 'хир', 'sometimes': 'самтаймз', 'watches': 'вочиз',
    'careers': 'карирз', 'beyond': 'бийонд', 'goes': 'гоуз', 'families': 'фэмилиз',
    'led': 'лед', 'museum': 'мьюзиам', 'themselves': 'зэмселвз', 'fan': 'фэн',
    'transport': 'транспорт', 'interesting': 'интерестинг', 'blogs': 'блогз',
    'wife': 'вайф', 'evaluation': 'и вэлюейшн', 'accepted': 'эксептид', 'former': 'формер',
    'implementation': 'имплиментейшн', 'ten': 'тен', 'hits': 'хитс', 'zone': 'зоун',
}

# Дополнительные ключевые слова Python (транскрипция для подстановки в слова)
_PYTHON_ONLY = {'def': 'деф', 'yield': 'йилд', 'pass': 'пас', 'break': 'брейк', 'continue': 'контину',
                'raise': 'рейз', 'try': 'трай', 'except': 'эксепт', 'finally': 'файнали', 'import': 'импорт',
                'global': 'глоубл', 'nonlocal': 'нонлоукал', 'lambda': 'лямбда', 'assert': 'ассерт',
                'del': 'дел', 'async': 'асинк', 'await': 'авейт', 'match': 'мэтч', 'case': 'кейс',
                'elif': 'элиф', 'with': 'виз', 'True': 'тру', 'False': 'фолс', 'None': 'нон'}
for k, v in _PYTHON_ONLY.items():
    EXPLICIT_ENGLISH.setdefault(k, v)


def _get_english_words_2000_path() -> str:
    """Путь к файлу со списком 2000 английских слов (рядом с данным модулем)."""
    return os.path.join(os.path.dirname(__file__), 'tts_english_words_2000.txt')


_ENGLISH_WORDS_2000_CACHE: Optional[Dict[str, str]] = None


def get_english_words_2000() -> Dict[str, str]:
    """
    Словарь: 2000 самых частых английских слов -> русская транскрипция.
    Список слов: tts_english_words_2000.txt (по одному на строку) или встроенный список.
    Результат кэшируется.
    """
    global _ENGLISH_WORDS_2000_CACHE
    if _ENGLISH_WORDS_2000_CACHE is not None:
        return _ENGLISH_WORDS_2000_CACHE
    path = _get_english_words_2000_path()
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            words = [w.strip().lower() for w in f if w.strip()]
    else:
        # Встроенный запас: 2000 слов из частотного списка (если файла нет)
        builtin = (
            "the of and to a in for is on that by this with i you it not or be are from at as your "
            "all have new more an was we will home can us about if page my has search free but our "
            "one other do no information time they site he up may what which their news out use any "
            "there see only so his when contact here business who web also now help get pm view online "
            "c e first am been would how were me s services some these click its like service x than "
            "find price date back top people had list name just over state year day into email two "
            "health n world re next used go b work last most products music buy data make them should "
            "product system post her city t add policy number such please available copyright support "
            "message after best software then jan good video well d where info rights public books "
            "high school through m each links she review years order very privacy book items company r "
            "read group sex need many user said de does set under general research university january "
            "mail full map reviews program life know games way days management p part could great united "
            "hotel real f item international center ebay must store travel comments made development report "
            "off member details line terms before hotels did send right type because local those using "
            "results office education national car design take posted internet address community within "
            "states area want phone dvd shipping reserved subject between forum family l long based w "
            "code show o even black check special prices website index being women much sign file link "
            "open today technology south case project same pages uk version section own found sports "
            "house related security both g county american photo game members power while care network "
            "down computer systems three total place end following download h him without per access "
            "think north resources current posts big media law control water history pictures size art "
            "personal since including guide shop directory board location change white text small rating "
            "rate government children during usa return students v shopping account times sites level "
            "digital profile previous form events love old john main call hours image department title "
            "description non k y insurance another why shall property class cd still money quality every "
            "listing content country private little visit save tools low reply customer december compare "
            "movies include college value article york man card jobs provide j food source author different "
            "press u learn sale around print course job canada process teen room stock training too "
            "credit point join science men categories advanced west sales look english left team estate "
            "box conditions select windows photos gay thread week category note live large gallery table "
            "register however june october november market library really action start series model "
            "features air industry plan human provided tv yes required second hot accessories cost movie "
            "forums march la september better say questions july yahoo going medical test friend come dec "
            "server pc study application cart staff articles san feedback again play looking issues april "
            "never users complete street topic comment financial things working against standard tax "
            "person below mobile less got blog party payment equipment login student let programs offers "
            "legal above recent park stores side act problem red give memory performance social q august "
            "quote language story sell options experience rates create key body young america important "
            "field few east paper single ii age activities club example girls additional password z "
            "latest something road gift question changes night ca hard texas oct pay four poker status "
            "browse issue range building seller court february always result audio light write war nov "
            "offer blue groups al easy given files event release analysis request fax china making "
            "picture needs possible might professional yet month major star areas future space committee "
            "hand sun cards problems london washington meeting rss become interest id child keep enter "
            "california porn share similar garden schools million added reference companies listed baby "
            "learning energy run delivery net popular term film stories put computers journal reports co "
            "try welcome central images president notice god original head radio until cell color self "
            "council away includes track australia discussion archive once others entertainment agreement "
            "format least society months log safety friends sure faq trade edition cars messages marketing "
            "tell further updated association able having provides david fun already green studies close "
            "common drive specific several gold feb living sep collection called short arts lot ask display "
            "limited powered solutions means director daily beach past natural whether due et electronics "
            "five upon period planning database says official weather mar land average done technical "
            "window france pro region island record direct microsoft conference environment records st "
            "district calendar costs style url front statement update parts aug ever downloads early miles "
            "sound resource present applications either ago document word works material bill apr written "
            "talk federal hosting rules final adult tickets thing centre requirements via cheap nude kids "
            "finance true minutes else mark third rock gifts europe reading topics bad individual tips plus "
            "auto cover usually edit together videos percent fast function fact unit getting global tech "
            "meet far economic en player projects lyrics often subscribe submit germany amount watch "
            "included feel though bank risk thanks everything deals various words linux jul production "
            "commercial james weight town heart advertising received choose treatment newsletter archives "
            "points knowledge magazine error camera jun girl currently construction toys registered clear "
            "golf receive domain methods chapter makes protection policies loan wide beauty manager india "
            "position taken sort listings models michael known half cases step engineering florida simple "
            "quick none wireless license paul friday lake whole annual published later basic sony shows "
            "corporate google church method purchase customers active response practice hardware figure "
            "materials fire holiday chat enough designed along among death writing speed html countries "
            "loss face brand discount higher effects created remember standards oil bit yellow political "
            "increase advertise kingdom base near environmental thought stuff french storage oh japan "
            "doing loans shoes entry stay nature orders availability africa summary turn mean growth notes "
            "agency king monday european activity copy although drug pics western income force cash "
            "employment overall bay river commission ad package contents seen players engine port album "
            "regional stop supplies started administration bar institute views plans double dog build "
            "screen exchange types soon sponsored lines electronic continue across benefits needed season "
            "apply someone held ny anything printer condition effective believe organization effect asked "
            "eur mind sunday selection casino pdf lost tour menu volume cross anyone mortgage hope silver "
            "corporation wish inside solution mature role rather weeks addition came supply nothing "
            "certain usr executive running lower necessary union jewelry according dc clothing mon com "
            "particular fine names robert homepage hour gas skills six bush islands advice career military "
            "rental decision leave british teens pre huge sat woman facilities zip bid kind sellers "
            "middle move cable opportunities taking values division coming tuesday object lesbian "
            "appropriate machine logo length actually nice score statistics client ok returns capital "
            "follow sample investment sent shown saturday christmas england culture band flash ms lead "
            "george choice went starting registration fri thursday courses consumer hi airport foreign "
            "artist outside furniture levels channel letter mode phones ideas wednesday structure fund "
            "summer allow degree contract button releases wed homes super male matter custom virginia "
            "almost took located multiple asian distribution editor inn industrial cause potential song "
            "cnet ltd los hp focus late fall featured idea rooms female responsible inc communications "
            "win associated thomas primary cancer numbers reason tool browser spring foundation answer "
            "voice eg friendly schedule documents communication purpose feature bed comes police everyone "
            "independent ip approach cameras brown physical operating hill maps medicine deal hold "
            "ratings chicago forms glass happy tue smith wanted developed thank safe unique survey prior "
            "telephone sport ready feed animal sources mexico population pa regular secure navigation "
            "operations therefore ass simply evidence station christian round paypal favorite understand "
            "option master valley recently probably thu rentals sea built publications blood cut worldwide "
            "improve connection publisher hall larger anti networks earth parents nokia impact transfer "
            "introduction kitchen strong tel carolina wedding properties hospital ground overview ship "
            "accommodation owners disease tx excellent paid italy perfect hair opportunity kit classic "
            "basis command cities william express anal award distance tree peter assessment ensure thus "
            "wall ie involved el extra especially interface pussy partners budget rated guides success "
            "maximum ma operation existing quite selected boy amazon patients restaurants beautiful "
            "warning wine locations horse vote forward flowers stars significant lists technologies owner "
            "retail animals useful directly manufacturer ways est son providing rule mac housing takes "
            "iii gmt bring catalog searches max trying mother authority considered told xml traffic "
            "programme joined input strategy feet agent valid bin modern senior ireland sexy teaching "
            "door grand testing trial charge units instead canadian cool normal wrote enterprise ships "
            "entire educational md leading metal positive fl fitness chinese opinion mb asia football "
            "abstract uses output funds mr greater likely develop employees artists alternative "
            "processing responsibility resolution java guest seems publication pass relations trust van "
            "contains session multi photography republic fees components vacation century academic "
            "assistance completed skin graphics indian prev ads mary il expected ring grade dating "
            "pacific mountain organizations pop filter mailing vehicle longer consider int northern behind "
            "panel floor german buying match proposed default require iraq boys outdoor deep morning "
            "otherwise allows rest protein plant reported hit transportation mm pool mini politics "
            "partner disclaimer authors boards faculty parties fish membership mission eye string sense "
            "modified pack released stage internal goods recommended born unless richard detailed "
            "japanese race approved background target except character usb maintenance ability maybe "
            "functions ed moving brands places php pretty trademarks phentermine spain southern yourself "
            "etc winter rape battery youth pressure submitted boston incest debt keywords medium "
            "television interested core break purposes throughout sets dance wood msn itself defined "
            "papers playing awards fee studio reader virtual device established answers rent las remote "
            "dark programming external apple le regarding instructions min offered theory enjoy remove "
            "aid surface minimum visual host variety teachers isbn martin manual block subjects agents "
            "increased repair fair civil steel understanding songs fixed wrong beginning hands associates "
            "finally az updates desktop classes paris ohio gets sector capacity requires jersey un fat "
            "fully father electric saw instruments quotes officer driver businesses dead respect unknown "
            "specified restaurant mike trip pst worth mi procedures poor teacher xxx eyes relationship "
            "workers farm fucking georgia peace traditional campus tom showing creative coast benefit "
            "progress funding devices lord grant sub agree fiction hear sometimes watches careers beyond "
            "goes families led museum themselves fan transport interesting blogs wife evaluation accepted "
            "former implementation ten hits zone"
        )
        words = builtin.split()[:2000]

    result: Dict[str, str] = {}
    for w in words:
        w = w.lower().strip()
        if not w or w in result:
            continue
        result[w] = EXPLICIT_ENGLISH.get(w, _simple_english_to_russian(w))
    # Добавляем ключевые слова Python, которых может не быть в частотном списке
    for k, v in _PYTHON_ONLY.items():
        result.setdefault(k.lower() if k not in ('True', 'False', 'None') else k, v)
    _ENGLISH_WORDS_2000_CACHE = result
    return result


def get_python_operator_symbols_sorted() -> list:
    """Список символьных операторов Python, отсортированный по убыванию длины (для подстановки)."""
    return _PYTHON_OP_SORTED
