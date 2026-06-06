import Foundation

/// Language tag → prompt slot index mapping shipped with multilingual Nemotron
/// bundles as `languages.json`. The model uses this to look up the one-hot
/// position in the `language_mask` encoder input (shape `[1, numPrompts]`).
///
/// File format (CoreML bundle):
/// ```json
/// {"promptDictionary": {"en-US": 0, "en": 0, ... "uk-UA": 19, "auto": 101}}
/// ```
public struct NemotronLanguages: Sendable, Codable {
    public let promptDictionary: [String: Int]

    public init(promptDictionary: [String: Int]) {
        self.promptDictionary = promptDictionary
    }

    public static func load(from url: URL) throws -> NemotronLanguages {
        let data = try Data(contentsOf: url)
        if let wrapped = try? JSONDecoder().decode(NemotronLanguages.self, from: data) {
            return wrapped
        }
        // Fall back to a bare dictionary (some bundles ship a flat map).
        let flat = try JSONDecoder().decode([String: Int].self, from: data)
        return NemotronLanguages(promptDictionary: flat)
    }

    /// English-only fallback used when no `languages.json` is present — the
    /// encoder of such bundles takes no `language_mask`, so the slot is moot.
    public static let englishOnly = NemotronLanguages(promptDictionary: ["auto": 0])

    /// Find the prompt slot for a language tag. Tries the full tag first
    /// (`uk-UA`), falls back to the language prefix (`uk`), then to `"auto"`.
    public func slot(for language: String?) -> Int {
        guard let language = language, !language.isEmpty else {
            return promptDictionary["auto"] ?? 0
        }
        if let s = promptDictionary[language] { return s }
        let normalized = language.replacingOccurrences(of: "_", with: "-")
        if let s = promptDictionary[normalized] { return s }
        let prefix = String(normalized.split(separator: "-").first ?? "")
        if !prefix.isEmpty, let s = promptDictionary[prefix] { return s }
        if !prefix.isEmpty, let s = promptDictionary[prefix.lowercased()] { return s }
        return promptDictionary["auto"] ?? 0
    }

    public var count: Int { promptDictionary.count }
}
