import json
import os

def load_config():
    # Load configuration from config.json file
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

def extract_data(file_path, default_weight):
    # Extract data from a JSON file located at file_path
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    trained_words = ",".join(data.get("trainedWords", []))
    weight = default_weight
    images = data.get("images", [])
    for image in images:
        meta = image.get("meta", {})
        if meta:
            resources = meta.get("resources", [])
            for resource in resources:
                if resource.get("type") == "lora":
                    weight = resource.get("weight", default_weight)
                    return trained_words, weight  # Exit early if weight is found
    return trained_words, weight

def main():
    config = load_config()
    input_path = config["input_path"]
    output_path = config["output_path"]
    output_format = config["output_format"]
    default_weight = config["default_weight"]
    with open(output_path, "w", encoding="utf-8") as f:
        if output_format == "json":
            f.write("[")  # Start JSON array
        file_names = (file_name for file_name in os.listdir(input_path) if file_name.endswith(".civitai.info"))
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(input_path, file_name)
            trained_words, weight = extract_data(file_path, default_weight)
            file_name_without_ext = os.path.splitext(file_name)[0]
            model_name = file_name_without_ext.replace(".civitai", "")
            result = f"<lora:{model_name}:{weight}>,{trained_words}"
            if output_format == "json":
                json.dump(result, f, ensure_ascii=False)
                if i != len(file_names) - 1:
                    f.write(",")
            else:
                f.write(result)
                f.write("\n")
        if output_format == "json":
            f.write("]")  # End JSON array

if __name__ == "__main__":
    main()