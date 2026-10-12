using System.Text.Json.Nodes;

namespace Pins.AI.Runtime;

public static class ToolPromptSchema {
    public const string TargetDescription = "Append a DSO container with Smart Exposure blocks to the loaded sequence; does not start it or create a blank root. Supply J2000 ra in hours, dec in degrees, and exposures OR template.";

    // Remove schema annotations only; retain argument names, enums and all validation constraints.
    public static JsonNode Compact(JsonNode node) {
        if (node is not JsonObject schema) return node.DeepClone();
        var result = new JsonObject();
        foreach (var (key, value) in schema) {
            if (key is "description" or "title" or "examples" or "$comment") continue;
            if (value is JsonObject map && key is "properties" or "$defs" or "definitions" or "patternProperties") {
                var copy = new JsonObject();
                foreach (var (name, definition) in map) copy[name] = definition is null ? null : Compact(definition);
                result[key] = copy;
            }
            else if (value is JsonArray choices && key is "anyOf" or "allOf" or "oneOf" or "prefixItems")
                result[key] = new JsonArray(choices.Select(v => v is null ? null : Compact(v)).ToArray());
            else if (key is "items" or "additionalProperties" or "contains" or "propertyNames" or "not" or "if" or "then" or "else")
                result[key] = value is null ? null : Compact(value);
            else result[key] = value?.DeepClone();
        }
        return result;
    }
}
