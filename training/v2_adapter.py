"""Source-verified adapter limits and localized synthetic sequence-result repairs."""
import re

from common import sha256

START_REPLIES = dict(zip(('en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'ja', 'zh'), (
    'PINS reports that the sequence is running. Its imaging work is not complete; monitor its status.',
    'PINS meldet, dass die Sequenz läuft. Die Aufnahmen sind noch nicht abgeschlossen; überwache den Status.',
    "PINS indique que la séquence est en cours. Les prises de vue ne sont pas terminées ; surveillez son état.",
    'PINS informa de que la secuencia está en marcha. Las capturas no han terminado; consulta su estado.',
    'PINS segnala che la sequenza è in esecuzione. Le acquisizioni non sono terminate; controlla lo stato.',
    'O PINS informa que a sequência está em execução. As capturas não terminaram; acompanhe o estado.',
    'PINS meldt dat de reeks loopt. De opnamen zijn nog niet voltooid; controleer de status.',
    'PINS zgłasza, że sekwencja działa. Zdjęcia nie są jeszcze ukończone; sprawdzaj jej stan.',
    'PINS によるとシーケンスは実行中です。撮影はまだ完了していません。状態を確認してください。',
    'PINS 报告序列正在运行。拍摄尚未完成，请继续查看状态。')))

STOP_REPLIES = dict(zip(('en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'ja', 'zh'), (
    'PINS accepted the sequence stop request. Verify the actual status before confirming it has stopped; this does not park the mount, stop guiding or warm the camera.',
    'PINS hat den Sequenzstopp angefordert. Bestätige den Stillstand erst nach einer Statusprüfung; Montierung, Guiding und Kamerawärmung sind separate Vorgänge.',
    "PINS a accepté la demande d’arrêt de la séquence. Vérifiez son état avant de confirmer l’arrêt ; cela ne parque pas la monture, n’arrête pas le guidage et ne réchauffe pas la caméra.",
    'PINS aceptó la solicitud de detener la secuencia. Comprueba el estado antes de confirmar la parada; esto no aparca la montura, detiene el guiado ni calienta la cámara.',
    'PINS ha accettato la richiesta di arresto della sequenza. Verifica lo stato prima di confermare; ciò non parcheggia la montatura, arresta la guida o riscalda la camera.',
    'O PINS aceitou o pedido de parar a sequência. Verifique o estado antes de confirmar; isso não estaciona a montagem, para a guiagem nem aquece a câmara.',
    'PINS heeft het stopverzoek voor de reeks geaccepteerd. Controleer de status voordat je de stop bevestigt; de montering parkeren, guiding stoppen en de camera opwarmen zijn aparte acties.',
    'PINS przyjął żądanie zatrzymania sekwencji. Sprawdź stan przed potwierdzeniem; nie oznacza to parkowania montażu, zatrzymania guidingu ani ogrzania kamery.',
    'PINS はシーケンスの停止要求を受け付けました。停止を断定する前に状態を確認してください。架台の格納、ガイド停止、カメラの昇温は別の操作です。',
    'PINS 已接受停止序列的请求。确认停止前请检查实际状态；此操作不会停放赤道仪、停止导星或升温相机。')))


def inspect_adapter(source, revision):
    """Fail on unfamiliar behavior rather than silently reusing this adapter profile."""
    relative = {
        'plan': 'NINA.Mcp.Adapter/Sequence/NinaSequenceService.Plan.cs',
        'editing': 'NINA.Mcp.Adapter/Sequence/NinaSequenceService.Editing.cs',
        'equipment': 'NINA.Mcp.Adapter/NinaObservatory.Equipment.cs',
        'status': 'NINA.Mcp.Adapter/NinaObservatory.Status.cs',
        'sequence': 'NINA.Mcp.Adapter/Sequence/NinaSequenceService.cs',
        'registration': 'NINA.Mcp.Adapter/McpAdapterServiceCollectionExtensions.cs',
        'host': 'NINA/Program.cs',
    }
    texts = {key: (source / name).read_text(encoding='utf-8-sig') for key, name in relative.items()}
    match = re.search(r'private const int MaxDitherEvery = (\d+);', texts['plan'])
    if not match or int(match[1]) != 32:
        raise ValueError('Exposure-plan dither limit changed; review policy and gateway before regeneration')
    anchors = {
        'editing': 'trimmed.EndsWith(".template", StringComparison.OrdinalIgnoreCase)',
        'equipment': 'Math.Abs(steps - Math.Round(steps)) > 1e-6',
        'sequence': 'Start requested; the sequencer is still preparing',
        'registration': 'services.AddSingleton<IPinsObservatory, NinaObservatory>();',
        'host': 'builder.Services.AddPinsMcpForNina(mcpOptions);',
    }
    for key, anchor in anchors.items():
        if anchor not in texts[key]:
            raise ValueError('Adapter source contract changed: ' + key)
    if 'app.MapPinsMcp();' not in texts['host']:
        raise ValueError('MCP host route mapping missing')
    return {'sourceRevision': revision, 'verification': 'source inspection; no live endpoint or equipment',
            'ditherEveryMaximum': 32, 'reservedSequenceSuffix': '.template',
            'switchStepTolerance': 1e-6, 'switchIndex': 'index in WritableSwitches; read-only switches have no writable index',
            'sequenceStart': 'returns when running, ended immediately, or still preparing; does not wait for imaging completion',
            'sequenceStop': 'requests cancellation including a preparing start; confirm actual status',
            'captureSave': 'save=true queues image saving; result FilePath may be null; file durability is not confirmed',
            'status': 'all eleven device slots; disconnected name null; readings may be null',
            'hostWiring': 'NinaObservatory registered and /mcp mapped in Program.cs; deployed compatibility unverified',
            'sourceHashes': {name: sha256(source / name) for name in relative.values()}}


def repair_feedback(row):
    repaired = []
    device_names = ('camera', 'mount', 'focuser', 'filter_wheel', 'guider', 'rotator',
                    'flat_device', 'dome', 'switch', 'weather', 'safety_monitor')
    import json
    for index, message in enumerate(row['messages']):
        if message['role'] != 'tool':
            continue
        payload = json.loads(message['content'])
        name = message['name']
        if name == 'get_status' and 'devices' in payload:
            by_name = {device['device']: device for device in payload['devices']}
            for device in by_name.values():
                if not device['connected']:
                    device['name'] = None
            payload['devices'] = [by_name.get(device, {'device': device, 'connected': False, 'name': None, 'properties': {}})
                                  for device in device_names]
            repaired.append('allStatusDeviceSlots')
        if name in ('start_sequence', 'stop_sequence') and payload.get('success') is True:
            if name == 'start_sequence':
                payload['message'] = 'The advanced sequence is running. Monitor it with get_status, get_sequence and get_recent_images.'
                reply = START_REPLIES[row['locale']]
            else:
                payload['message'] = 'Stop requested. The running instruction is cancelled; the sequence ends within a few seconds.'
                reply = STOP_REPLIES[row['locale']]
            if index + 1 < len(row['messages']) and row['messages'][index + 1]['role'] == 'assistant' and not row['messages'][index + 1].get('tool_calls'):
                row['messages'][index + 1]['content'] = reply
            repaired.append(name + 'CompletionSemantics')
        message['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    return repaired
