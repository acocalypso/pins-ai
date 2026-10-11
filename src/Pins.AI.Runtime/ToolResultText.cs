using System.Text.Json.Nodes;

namespace Pins.AI.Runtime;

// Fixed user-facing state descriptions. No equipment prose is executed or
// echoed as an instruction; raw diagnostic payload remains marked untrusted.
public static class ToolResultText {
    private static readonly Dictionary<string, string[]> Text = new() {
        ["en"] = ["The operation failed or was cancelled; check equipment status.", "Stop requested; verify equipment status.", "The operation is still running.", "The job completed successfully.", "Completion is not confirmed."],
        ["de"] = ["Der Vorgang ist fehlgeschlagen oder wurde abgebrochen; Gerätestatus prüfen.", "Stopp angefordert; Gerätestatus prüfen.", "Der Vorgang läuft noch.", "Der Auftrag wurde erfolgreich abgeschlossen.", "Der Abschluss ist nicht bestätigt."],
        ["fr"] = ["L’opération a échoué ou a été annulée ; vérifiez l’état du matériel.", "Arrêt demandé ; vérifiez l’état du matériel.", "L’opération est encore en cours.", "La tâche s’est terminée avec succès.", "La fin de l’opération n’est pas confirmée."],
        ["es"] = ["La operación falló o se canceló; comprueba el estado del equipo.", "Parada solicitada; comprueba el estado del equipo.", "La operación sigue en curso.", "La tarea terminó correctamente.", "No se ha confirmado la finalización."],
        ["it"] = ["L’operazione è fallita o è stata annullata; verifica lo stato dei dispositivi.", "Arresto richiesto; verifica lo stato dei dispositivi.", "L’operazione è ancora in corso.", "Il lavoro è stato completato correttamente.", "Il completamento non è confermato."],
        ["pt"] = ["A operação falhou ou foi cancelada; verifica o estado do equipamento.", "Paragem solicitada; verifica o estado do equipamento.", "A operação ainda está em curso.", "A tarefa foi concluída com sucesso.", "A conclusão não está confirmada."],
        ["nl"] = ["De handeling is mislukt of geannuleerd; controleer de apparaatstatus.", "Stop aangevraagd; controleer de apparaatstatus.", "De handeling loopt nog.", "De taak is succesvol afgerond.", "Voltooiing is niet bevestigd."],
        ["pl"] = ["Operacja nie powiodła się lub została anulowana; sprawdź stan urządzeń.", "Zażądano zatrzymania; sprawdź stan urządzeń.", "Operacja nadal trwa.", "Zadanie zakończyło się pomyślnie.", "Zakończenie nie jest potwierdzone."],
        ["ja"] = ["操作が失敗またはキャンセルされました。機器の状態を確認してください。", "停止を要求しました。機器の状態を確認してください。", "操作はまだ実行中です。", "ジョブは正常に完了しました。", "完了は確認されていません。"],
        ["zh"] = ["操作失败或已取消；请检查设备状态。", "已请求停止；请检查设备状态。", "操作仍在运行。", "任务已成功完成。", "尚未确认完成。"]
    };

    public static string Describe(JsonObject result, string locale) {
        var language = locale.Split('-', '_')[0];
        var messages = Text.GetValueOrDefault(language, Text["en"]);
        var index = result["isError"]?.GetValue<bool>() == true ? 0 :
            result["stopRequested"]?.GetValue<bool>() == true ? 1 :
            result["requiresPolling"]?.GetValue<bool>() == true ? 2 :
            result["completionConfirmed"]?.GetValue<bool>() == true ? 3 : 4;
        return messages[index];
    }
}
