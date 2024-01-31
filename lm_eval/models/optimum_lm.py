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
    inplen = None
    """
    if not find_spec("optimum"):
        raise Exception(
            "package `optimum` is not installed. Please install it via `pip install optimum[openvino]`"
        )
    else:
        from optimum.intel.openvino import OVModelForCausalLM
    """

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

        revision = revision + ("/" + subfolder if subfolder is not None else "")
        
        self._get_config(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        # determine which of 'causal' and 'seq2seq' backends to use
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )
        self._backend = backend

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=False,
        **kwargs,
    ) -> None:
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
        #config,
        backend="default",
        trust_remote_code=False,
    ) -> None:
        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.AUTO_MODEL_CLASS = OVModelForCausalLM
            elif backend == "seq2seq":
                self.AUTO_MODEL_CLASS = OVModelForSeq2SeqLM
            eval_logger.info(
                f"Overrode Optimum model backend type, and using type '{backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
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
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to AutoModelForCausalLM
                self.AUTO_MODEL_CLASS = OVModelForCausalLM
       
        assert self.AUTO_MODEL_CLASS in [
            OVModelForCausalLM,
            OVModelForSeq2SeqLM,
        ]
        return None

