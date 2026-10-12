using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Pins.AI.Runtime;

/// <summary>Retrieves a small tool context; the model chooses the call and the gateway authorizes it.</summary>
public static class ChatToolRouter {
    // Whole words for actions avoid treating "connected?" as a connect instruction.
    // Aliases supplement live names; they retrieve candidates, never prescribe a call.
    private static readonly (string Name, string Pattern)[] Intents = [
        ("get_status", @"\b(status|equipment|connected|observatory|temperature|weather|gerat\w*|verbunden|angeschlossen|temperatur|wetter|ausrustung)\b"),
        ("get_recent_images", @"\b(recent images|last image|image history|latest images|hfr|letzte\w* bild\w*|bildhistorie|sternanzahl)\b"),
        ("get_sequence", @"\b(show|inspect|current|progress|zeige|aktuell\w*|fortschritt)\b.*\b(sequence|sequenz)\b"),
        ("list_saved_sequences", @"\b(saved sequences|list sequences|gespeicherte\w* sequenz\w*)\b"),
        ("list_target_templates", @"\b(template\w*|vorlage\w*)\b"),
        ("get_job", @"\b(job[-_][\w-]+|job status|is it done|has it finished|ist es fertig)\b"),
        ("list_jobs", @"\b(jobs|background|hintergrund|auftrage)\b"),
        ("cancel_job", @"\b(cancel|abort|abbrechen|abbruch)\b.*\b(job|auftrag)\b"),
        ("stop_all", @"\b(emergency|stop everything|stop all|abort everything|notfall|notstopp|alles stoppen|stoppe alles)\b"),
        ("connect_device", @"\b(connect|verbinde|verbinden|anschlie.en)\b"),
        ("disconnect_device", @"\b(disconnect|trenne|trennen)\b"),
        ("slew_mount", @"\b(slew|goto|go to|schwenke|anfahren)\b"),
        ("park_mount", @"\b(park|parke|parken)\b"),
        ("unpark_mount", @"\b(unpark|ausparken|entparke)\b"),
        ("home_mount", @"\b(home mount|home position|referenzfahrt|referenzposition)\b"),
        ("set_tracking", @"\b(tracking|nachfuhrung)\b"),
        ("capture_image", @"\b(capture|take.*(?:image|exposure|photo)|aufnahme|belichte|fotografiere)\b"),
        ("cool_camera", @"\b(cool|cooling|kuhle|kuhlen)\b"),
        ("warm_camera", @"\b(warm|warming|aufwarmen|aufwarme)\b"),
        ("move_focuser", @"\b(move.*focuser|focuser.*position|fokussierer.*(?:bewege|position)|bewege.*fokussierer)\b"),
        ("run_autofocus", @"\b(autofocus|autofokus|focus automatically)\b"),
        ("change_filter", @"\b(change.*filter|select.*filter|filter.*wechsel|wechsle.*filter)\b"),
        ("start_guiding", @"\b(start|begin|starte)\b.*\b(guiding|autoguiding)\b"),
        ("stop_guiding", @"\b(stop|stoppe|beende)\b.*\b(guiding|autoguiding)\b"),
        ("set_switch", @"\b(switch|relay|schalter|relais)\b"),
        ("load_sequence", @"\b(load|lade|laden)\b.*\b(sequence|sequenz)\b"),
        ("add_target", @"\b(add.*target|fuge.*ziel|ziel.*hinzu)\b"),
        ("clear_targets", @"\b(clear.*targets|remove.*targets|losche.*ziele|entferne.*ziele)\b"),
        ("save_sequence", @"\b(save|speichere|speichern)\b.*\b(sequence|sequenz)\b"),
        ("start_sequence", @"\b(start|resume|starte|fortsetzen)\b.*\b(sequence|sequenz)\b"),
        ("stop_sequence", @"\b(stop|stoppe|beende)\b.*\b(sequence|sequenz)\b"),
        ("reset_sequence", @"\b(reset|zurucksetzen|setze.*zuruck)\b.*\b(sequence|sequenz)\b")
    ];
    private static readonly Dictionary<string, string> Localized = new(StringComparer.Ordinal) {
        ["get_status"] = @"\b(equipement\w*|connecte\w*|conectad\w*|attrezzatura|collegat\w*|equipamento\w*|equipo|estado|stato|dispositivi|verbonden|apparatuur|apparaatstatus|sprzet|polaczon\w*)\b|機器|接続|设备|設備|连接|連接",
        ["get_recent_images"] = @"\b(images recentes|dernier\w* image|imagenes recientes|ultim\w* immagin\w*|imagens recentes|laatste beelden|ostatnie zdjecia)\b|最新.*画像|最近.*图像",
        ["get_sequence"] = @"\b(sequence actuelle|secuencia actual|sequenza attuale|sequencia atual|huidige sequentie|biezaca sekwencja)\b|現在.*シーケンス|当前.*序列",
        ["list_saved_sequences"] = @"\b(sequences enregistrees|secuencias guardadas|sequenze salvate|sequencias salvas|opgeslagen sequenties|zapisane sekwencje)\b|保存.*シーケンス|已保存.*序列",
        ["list_target_templates"] = @"\b(modeles de cible|plantillas|modelli di destinazione|modelos de alvo|sjablonen|szablony)\b|テンプレート|模板",
        ["get_job"] = @"\b(etat.*tache|estado.*trabajo|stato.*lavoro|estado.*tarefa|status.*taak|stan.*zadania)\b|ジョブ.*状態|任务.*状态",
        ["list_jobs"] = @"\b(taches|trabajos|lavori|tarefas|taken|zadania)\b|ジョブ一覧|任务列表",
        ["cancel_job"] = @"\b(annul\w*.*tache|cancel\w*.*trabajo|annull\w*.*lavoro|cancel\w*.*tarefa|annuleer.*taak|anuluj.*zadanie)\b|ジョブ.*キャンセル|取消.*任务",
        ["stop_all"] = @"\b(arret d'urgence|arrete tout|deten todo|ferma tutto|pare tudo|stop alles|zatrzymaj wszystko)\b|緊急停止|全部停止|紧急停止",
        ["connect_device"] = @"\b(connecter|connecte|conecta|conecte|conectar|connetti|collega|verbinden|verbind|polacz)\b|接続して|连接.*(?:相机|望远镜)",
        ["disconnect_device"] = @"\b(deconnect\w*|desconect\w*|disconnetti|scollega|verbreek|rozlacz)\b|切断して|断开",
        ["slew_mount"] = @"\b(pointer|apunta|puntare|girar|richt.*montering|skieruj)\b|導入して|转向",
        ["park_mount"] = @"\b(parquer|aparca|parcheggia|estacionar|parkeer|zaparkuj)\b|パークして|停放",
        ["unpark_mount"] = @"\b(deparquer|desaparca|sblocca.*montatura|desestacionar|ontpark|odparkuj)\b|パーク解除|解除停放",
        ["home_mount"] = @"\b(position initiale|posicion inicial|posizione iniziale|posicao inicial|thuispositie|pozycja poczatkowa)\b|ホーム位置|原点",
        ["set_tracking"] = @"\b(suivi|seguimiento|inseguimento|rastreamento|volgen|sledzenie)\b|追尾|跟踪",
        ["capture_image"] = @"\b(prends.*photo|capturer|captura|scatta|fotografa|maak.*foto|zrob.*zdjecie)\b|撮影して|拍摄|拍照",
        ["cool_camera"] = @"\b(refroid\w*|enfria\w*|raffredd\w*|resfria\w*|koel|schlodz)\b|冷却して|制冷|冷却相机",
        ["warm_camera"] = @"\b(rechauff\w*|calienta\w*|riscald\w*|aqueca|verwarm|ogrzej)\b|温めて|升温",
        ["move_focuser"] = @"\b(deplace.*focalis\w*|mueve.*enfocador|muovi.*focheggiatore|mova.*focador|verplaats.*focuser|przesun.*fokuser)\b|フォーカサー.*移動|移动.*调焦",
        ["run_autofocus"] = @"\b(autofocus|autoenfoque|autofoco|automatyczn\w*.*ostrosc)\b|オートフォーカス|自动对焦",
        ["change_filter"] = @"\b(change.*filtre|cambia.*filtro|muda.*filtro|wissel.*filter|zmien.*filtr)\b|フィルター.*変更|切换.*滤镜",
        ["start_guiding"] = @"\b(demarr\w*.*guidage|inicia.*guiado|avvia.*guida|inici\w*.*guiagem|start.*volgen|rozpocznij.*prowadzenie)\b|ガイド.*開始|开始.*导星",
        ["stop_guiding"] = @"\b(arret\w*.*guidage|deten.*guiado|ferma.*guida|pare.*guiagem|stop.*volgen|zatrzymaj.*prowadzenie)\b|ガイド.*停止|停止.*导星",
        ["set_switch"] = @"\b(interrupteur|interruptor|interruttore|schakelaar|przelacznik)\b|スイッチ|开关",
        ["load_sequence"] = @"\b(charg\w*.*sequence|carg\w*.*secuencia|caric\w*.*sequenza|carreg\w*.*sequencia|laad.*sequentie|wczytaj.*sekwencje)\b|シーケンス.*読み込|加载.*序列",
        ["add_target"] = @"\b(ajout\w*.*cible|anad\w*.*objetivo|aggiung\w*.*obiettivo|adicion\w*.*alvo|voeg.*doel|dodaj.*cel)\b|対象.*追加|添加.*目标",
        ["clear_targets"] = @"\b(supprim\w*.*cibles|elimin\w*.*objetivos|rimuov\w*.*obiettivi|remov\w*.*alvos|verwijder.*doelen|usun.*cele)\b|対象.*削除|清除.*目标",
        ["save_sequence"] = @"\b(enregistr\w*.*sequence|guard\w*.*secuencia|salv\w*.*sequenza|salv\w*.*sequencia|sla.*sequentie|zapisz.*sekwencje)\b|シーケンス.*保存|保存.*序列",
        ["start_sequence"] = @"\b(demarr\w*.*sequence|inici\w*.*secuencia|avvi\w*.*sequenza|inici\w*.*sequencia|start.*sequentie|rozpocznij.*sekwencje)\b|シーケンス.*開始|开始.*序列",
        ["stop_sequence"] = @"\b(arret\w*.*sequence|deten.*secuencia|ferma.*sequenza|pare.*sequencia|stop.*sequentie|zatrzymaj.*sekwencje)\b|シーケンス.*停止|停止.*序列",
        ["reset_sequence"] = @"\b(reinitialis\w*.*sequence|reinici\w*.*secuencia|ripristin\w*.*sequenza|reinici\w*.*sequencia|reset.*sequentie|zresetuj.*sekwencje)\b|シーケンス.*リセット|重置.*序列"
    };

    public static string[] Select(string message, JsonArray catalog, string[]? clarificationTools = null) {
        var text = Normalize(message);
        var scores = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var tool in catalog) {
            var name = tool!["name"]!.GetValue<string>();
            if (Regex.IsMatch(text, @"(?<!\w)" + Regex.Escape(name) + @"(?!\w)", RegexOptions.CultureInvariant)) scores[name] = 100;
            foreach (var intent in Intents.Where(i => i.Name == name))
                if (Regex.IsMatch(text, intent.Pattern, RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100)))
                    scores[name] = Math.Max(scores.GetValueOrDefault(name), name == "get_status" ? 5 : 10);
            if (Localized.TryGetValue(name, out var pattern) && Regex.IsMatch(text, Normalize(pattern), RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100)))
                scores[name] = Math.Max(scores.GetValueOrDefault(name), name == "get_status" ? 5 : 10);
        }
        // Unrecognized requests get no tool context and can receive text/clarification.
        // Retrieval cannot authorize an operation or invent a live tool.
        if (!scores.Values.Any(score => score > 5) && clarificationTools is not null) {
            foreach (var name in clarificationTools.Where(name => catalog.Any(t => t?["name"]?.GetValue<string>() == name)))
                scores[name] = Math.Max(scores.GetValueOrDefault(name), 10);
        }
        return scores.OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal).Take(4).Select(p => p.Key).ToArray();
    }

    private static string Normalize(string value) => string.Concat(value.ToLowerInvariant().Normalize(NormalizationForm.FormD)
        .Where(c => CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)).Normalize(NormalizationForm.FormC);
}
