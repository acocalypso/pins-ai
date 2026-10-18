using System.Globalization;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Pins.AI.Runtime;

/// <summary>Conservative origin check for model-produced coordinates, not proof of intent or authorization.</summary>
public static class CoordinateEvidence {
    private const string Number = @"([+-]?\d+(?:[.,]\d+)?)";
    public static void Validate(string name, JsonObject arguments, JsonArray history) {
        if (name is not ("slew_mount" or "center_target" or "add_target" or "plan_targets")) return;
        var known = new List<(double Ra, double Dec)>();
        foreach (var item in history) {
            if (item?["role"]?.GetValue<string>() == "user") {
                var text = item["content"]!.GetValue<string>();
                var ra = Regex.Matches(text, @"(?:(?<![\p{L}\p{N}])(?:RA|AR|right ascension|Rektaszension)|赤経|赤经)\s*[:=]?\s*" + Number + @"\s*(hours?|h\b|Stunden|heures?|ore|horas?|uur|godzin\w*|時間|小时|degrees?|Grad|degrés|gradi|graus|graden|stopni|度)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100));
                var dec = Regex.Matches(text, @"(?:(?<![\p{L}\p{N}])(?:Dec|declination|Deklination|déclinaison|declinazione|declinação|declinación|deklinacja|declinatie)|赤緯|赤纬)\s*[:=]?\s*" + Number, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100));
                if (ra.Count != dec.Count) continue;
                for (var i = 0; i < ra.Count; i++) {
                    var value = double.Parse(ra[i].Groups[1].Value.Replace(',', '.'), CultureInfo.InvariantCulture);
                    var unit = ra[i].Groups[2].Value.ToLowerInvariant();
                    if (Regex.IsMatch(unit, @"degree|grad|degré|gradi|graus|stopni|度", RegexOptions.CultureInvariant)) value /= 15;
                    known.Add((value, double.Parse(dec[i].Groups[1].Value.Replace(',', '.'), CultureInfo.InvariantCulture)));
                }
            }
            // Only structured results recorded by the server, never arbitrary prose
            // or model/assistant messages, may supply a single resolved catalog match.
            if (item?["role"]?.GetValue<string>() == "tool" && item["name"]?.GetValue<string>() == "find_targets") {
                var payload = JsonNode.Parse(item["content"]!.GetValue<string>());
                if (payload?["isError"]?.GetValue<bool>() == true) continue;
                if (payload?["objects"] is JsonArray objects && objects.Count == 1 &&
                    objects[0]?["raHours"] is JsonValue r && r.TryGetValue<double>(out var ra) &&
                    objects[0]?["decDegrees"] is JsonValue d && d.TryGetValue<double>(out var dec)) known.Add((ra, dec));
            }
        }
        var targets = name == "plan_targets" ? arguments["targets"]!.AsArray().Select(t => t!.AsObject()) : [arguments];
        foreach (var target in targets) {
            var ra = target["ra"]!.GetValue<double>();
            var dec = target["dec"]!.GetValue<double>();
            if (!known.Any(p => Math.Abs(p.Ra - ra) < 1e-8 && Math.Abs(p.Dec - dec) < 1e-8))
                throw new ArgumentException("Coordinate origin could not be established. Supply explicit J2000 RA with hours/degrees and Dec, or resolve one catalog match. No action was proposed.");
        }
    }
}
