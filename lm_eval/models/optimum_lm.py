import transformers
import optimum
from optimum.intel.openvino import OVModelForCausalLM, OVModelForSeq2SeqLM

from importlib.util import find_spec
from pathlib import Path

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from lm_eval import utils
eval_logger = utils.eval_logger

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

@register_model("openvino")
class OptimumLM(HFLM):
    """
    Optimum Intel provides a simple interface to optimize Transformer models and convert them to \
    OpenVINO™ Intermediate Representation (IR) format to accelerate end-to-end pipelines on \
    Intel® architectures using OpenVINO™ runtime.
    """
    AUTO_MODEL_CLASS = None

    def __init__(
        self,
        pretrained="gpt2",
        backend="default",
        revision="main",
        subfolder=None,
        device="cpu",
        trust_remote_code=False,
        **kwargs,
    ) -> None:

        self.openvino_device = device

        super().__init__(
            device=self.openvino_device,
            backend=kwargs.get("backend", "causal"),
            **kwargs,
        )

        #revision = revision + ("/" + subfolder if subfolder is not None else "")
        
        self._get_config(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        print("\n\nOptimum done with _get_config")
        
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )
        print("\n\nOptimum done with _get_backend")
        #self._backend = backend

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=False,
        **kwargs,
    ) -> None:
        print("\n\nWelcome to Optimum _create_model method.....")
        model_kwargs = kwargs if kwargs else {}
        model_file = Path(pretrained) / "openvino_model.xml"
        if model_file.exists():
            export = False
        else:
            export = True
        kwargs["ov_config"] = {"PERFORMANCE_HINT": "LATENCY","NUM_STREAMS": "1", "CACHE_DIR": "",}
        
        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            export=export,
            device=self.openvino_device.upper(),
            **model_kwargs,
        )

    def _get_backend(
        self,
        config: transformers.AutoConfig,
        backend="default",
        trust_remote_code=False,
    ) -> None:
        print("\n\nWelcome to Optimum _get_backend method.....")
        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            if backend == "causal":
                self.AUTO_MODEL_CLASS = OVModelForCausalLM
            elif backend == "seq2seq":
                self.AUTO_MODEL_CLASS = OVModelForSeq2SeqLM
            eval_logger.info(
                f"Overrode Optimum model backend type, and using type '{backend}'"
            )
        else:
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                self.AUTO_MODEL_CLASS = OVModelForSeq2SeqLM
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.AUTO_MODEL_CLASS = OVModelForCausalLM
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                    )
                self.AUTO_MODEL_CLASS = OVModelForCausalLM
       
        assert self.AUTO_MODEL_CLASS in [
            OVModelForCausalLM,
            OVModelForSeq2SeqLM,
        ]
        return None

