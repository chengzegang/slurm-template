use regex::Regex;
use walkdir::WalkDir;

pub fn listdir_rs(root: &str, pattern: Option<&str>, recursive: Option<bool>) -> Vec<String> {
    let mut result = Vec::new();
    let pattern_re = match pattern {
        Some(pattern) => pattern,
        None => "",
    };

    let mut walker = WalkDir::new(root);
    if !recursive.unwrap_or(false) {
        walker = walker.max_depth(1);
    }

    let re = Regex::new(pattern_re).unwrap();
    for entry in walker {
        let entry = entry.unwrap();
        let path = entry.path();
        let path_str = path.to_str().unwrap();
        if re.is_match(path_str) {
            result.push(path_str.to_string());
        }
    }
    result
}
