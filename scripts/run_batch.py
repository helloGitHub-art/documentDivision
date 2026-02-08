import argparse
from pathlib import Path

from src.sam_canny import batch_process_images


def _resolve_default_paths(project_root: Path):
    preferred_input = project_root / "data" / "test-images"
    legacy_input = project_root / "test" / "test-image"
    input_dir = preferred_input if preferred_input.exists() else legacy_input

    preferred_model = project_root / "models" / "sam" / "sam_vit_b_01ec64.pth"
    legacy_model = project_root / "sam_weights" / "sam_vit_b_01ec64.pth"
    model_path = preferred_model if preferred_model.exists() else legacy_model

    output_dir = project_root / "outputs" / "sam_canny"
    return input_dir, output_dir, model_path


def _build_arg_parser(default_input, default_output, default_model):
    parser = argparse.ArgumentParser(description="SAM + Canny 批量处理入口")
    parser.add_argument("--input-dir", default=str(default_input), help="输入图片目录")
    parser.add_argument("--output-dir", default=str(default_output), help="输出结果目录")
    parser.add_argument("--model-path", default=str(default_model), help="SAM权重文件路径")
    parser.add_argument("--model-type", default="vit_b", help="SAM模型类型")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny低阈值")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny高阈值")
    parser.add_argument("--show-process", action="store_true", help="显示中间处理结果")
    return parser


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_input, default_output, default_model = _resolve_default_paths(project_root)
    parser = _build_arg_parser(default_input, default_output, default_model)
    args = parser.parse_args()

    batch_process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sam_model_path=args.model_path,
        model_type=args.model_type,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        show_process=args.show_process,
        save_all=True,
    )


if __name__ == "__main__":
    main()
