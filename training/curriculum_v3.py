"""Deterministic train-only contrasts; never read test, eval_gold or hard_gold."""
import copy
import importlib.util
from pathlib import Path

from common import read_jsonl, fingerprint
from v2_contracts import PACKAGE, POLICY, definitions, validate_call

PROFILE = 'v3-contrastive'

# Additional equipment labels; synthetic translations still need native review.
EXTRA_LABELS = {
    'en': ['filter wheel', 'rotator', 'flat panel', 'dome', 'switch hub', 'weather monitor', 'safety monitor'],
    'de': ['Filterrad', 'Rotator', 'Flatfield-Panel', 'Kuppel', 'Schaltbox', 'Wettermonitor', 'Sicherheitsmonitor'],
    'fr': ['la roue à filtres', 'le rotateur', 'le panneau de flats', 'le dôme', 'le boîtier de commutation', 'la station météo', 'le moniteur de sécurité'],
    'es': ['la rueda de filtros', 'el rotador', 'el panel de flats', 'la cúpula', 'el concentrador de interruptores', 'la estación meteorológica', 'el monitor de seguridad'],
    'it': ['la ruota portafiltri', 'il rotatore', 'il pannello flat', 'la cupola', 'il gruppo di interruttori', 'la stazione meteo', 'il monitor di sicurezza'],
    'pt': ['a roda de filtros', 'o rotador', 'o painel de flats', 'a cúpula', 'o módulo de interruptores', 'a estação meteorológica', 'o monitor de segurança'],
    'nl': ['het filterwiel', 'de rotator', 'het flatpaneel', 'de koepel', 'de schakelhub', 'het weerstation', 'de veiligheidsmonitor'],
    'pl': ['koło filtrowe', 'rotator', 'panel flat', 'kopułę', 'moduł przełączników', 'stację pogodową', 'monitor bezpieczeństwa'],
    'ja': ['フィルターホイール', 'ローテーター', 'フラットパネル', 'ドーム', 'スイッチハブ', '気象モニター', '安全モニター'],
    'zh': ['滤镜轮', '旋转器', '平场面板', '圆顶', '开关集线器', '气象监测设备', '安全监测设备'],
}
CORRECTION = dict(zip(EXTRA_LABELS, [
    'Correction: the device is {label}.', 'Korrektur: Das Gerät ist {label}.',
    'Correction : il s’agit de {label}.', 'Corrección: el dispositivo es {label}.',
    'Correzione: il dispositivo è {label}.', 'Correção: o dispositivo é {label}.',
    'Correctie: het apparaat is {label}.', 'Poprawka: urządzenie to {label}.',
    '訂正：対象の機器は{label}です。', '更正：目标设备是{label}。']))
ASK = dict(zip(EXTRA_LABELS, [
    'Which device do you mean?', 'Welches Gerät meinst du?', 'Quel appareil veux-tu dire ?',
    '¿A qué dispositivo te refieres?', 'A quale dispositivo ti riferisci?', 'A que dispositivo te referes?',
    'Welk apparaat bedoel je?', 'Które urządzenie masz na myśli?', 'どの機器を指していますか？', '你指的是哪个设备？']))


def language_source():
    path = PACKAGE / 'scripts/build_dataset.py'
    spec = importlib.util.spec_from_file_location('pins_training_phrases', path)
    module = importlib.util.module_from_spec(spec)
    # The generator imports its sibling tool_catalog; it does not run main here.
    import sys
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def curriculum_rows(split):
    if split not in ('train', 'validation'):
        raise ValueError('Curriculum accepts only train/validation; gold is never a training source.')
    rows = read_jsonl(PACKAGE / 'data/audited' / f'{split}.jsonl')
    if split == 'validation':
        return rows  # Existing validation is for selection, never augmentation seeds.
    source = language_source()
    catalog = definitions()
    devices = catalog['connect_device']['inputSchema']['properties']['device']['enum']
    extras = ('filter_wheel', 'rotator', 'flat_device', 'dome', 'switch', 'weather', 'safety_monitor')
    added = []

    def row(locale, device, action, kind, text, approval=True, history=None):
        arguments = {'device': device}
        validate_call(action, arguments, catalog)
        context = POLICY
        if approval:
            context += ' Trusted gateway authorization for this turn (provided out of band; server must re-check): ' + action + '.'
        names = [action, 'disconnect_device' if action == 'connect_device' else 'connect_device', 'get_status']
        messages = [{'role': 'system', 'content': context}]
        messages += history or []
        messages += [{'role': 'user', 'content': text}]
        if approval:
            messages += [{'role': 'assistant', 'content': '', 'tool_calls': [
                {'type': 'function', 'function': {'name': action, 'arguments': arguments}}]}]
        else:
            # This is an authorization fixture, not an actual operator approval.
            messages += [{'role': 'assistant', 'content': source.CONFIRM[locale]}]
        group = f'curriculum-device-{device}-{action}-{kind}'
        added.append({'id': group + '-' + locale, 'group': group, 'locale': locale,
            'category': 'device_contrast' if approval else 'no_approval',
            'tools': [{'name': n, 'description': catalog[n]['description'], 'parameters': catalog[n]['inputSchema']} for n in names],
            'messages': messages, 'expected': {'first_tool': action if approval else None,
                                              'arguments': arguments if approval else None},
            'provenance': {'source': 'synthetic train-only contrast curriculum',
                           'hardwareExecution': False, 'linguisticReview': 'pending'}})

    for locale in EXTRA_LABELS:
        labels = {**source.DEVICE_NAMES[locale], **dict(zip(extras, EXTRA_LABELS[locale]))}
        for device in devices:
            label = labels[device]
            for action in ('connect_device', 'disconnect_device'):
                phrase = source.PHRASES[locale][action].format(device_text=label)
                for variant in ((0, 1, 3) if device in source.DEVICE_NAMES[locale] else (0,)):
                    row(locale, device, action, f'wording{variant}', source.USER_VARIANTS[locale][variant](phrase))
                row(locale, device, action, 'identifier', source.PHRASES[locale][action].format(device_text=f'{label} ({device})'))
                # A mistaken/underspecified earlier request must not dominate a correction.
                history = [{'role': 'user', 'content': source.PHRASES[locale][action].format(device_text='?')},
                           {'role': 'assistant', 'content': ASK[locale]}]
                row(locale, device, action, 'correction', CORRECTION[locale].format(label=label), history=history)
                row(locale, device, action, 'unapproved', phrase + source.NO_APPROVAL_SUFFIX[locale], approval=False)

    # Same training requests with competing tools: teach selection, not only copying
    # the sole exposed tool. Avoid large schemas so the Pi context budget stays small.
    simple = {'get_status', 'get_recent_images', 'list_jobs', 'get_job', 'connect_device',
              'disconnect_device', 'park_mount', 'unpark_mount', 'home_mount', 'start_guiding', 'stop_guiding'}
    for original in rows:
        if original['category'] != 'tool_success':
            continue
        name = original['expected'].get('first_tool')
        item = copy.deepcopy(original)
        item['id'] += '-competing-tools'
        # Keep the original scenario group so the split checker can detect leakage.
        names = {t['name'] for t in item['tools']}
        for candidate in (('get_status', 'connect_device', 'disconnect_device') if name in simple else ('get_status',)):
            if candidate not in names:
                item['tools'].append({'name': candidate, 'description': catalog[candidate]['description'],
                                      'parameters': catalog[candidate]['inputSchema']})
        item['tools'].sort(key=lambda t: fingerprint(item['id'] + t['name']))
        added.append(item)
    if len({r['id'] for r in rows + added}) != len(rows + added):
        raise ValueError('Duplicate curriculum identifiers')
    return rows + added
