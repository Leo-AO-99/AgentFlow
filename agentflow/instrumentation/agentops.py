import logging
import multiprocessing
import signal
import socket
import time

import flask
import setproctitle

logger = logging.getLogger(__name__)

# Module-level storage for originals
_original_handle_chat_attributes = None
_original_handle_response = None


def _patch_new_agentops():
    import agentops.instrumentation.providers.openai.wrappers.chat
    import agentops.instrumentation.providers.openai.stream_wrapper
    from agentops.instrumentation.providers.openai.wrappers.chat import handle_chat_attributes

    global _original_handle_chat_attributes

    if _original_handle_chat_attributes is not None:
        logger.warning("AgentOps already patched. Skipping.")
        return True

    _original_handle_chat_attributes = handle_chat_attributes

    def _handle_chat_attributes_with_tokens(args=None, kwargs=None, return_value=None, **kws):
        attributes = _original_handle_chat_attributes(args=args, kwargs=kwargs, return_value=return_value, **kws)

        prompt_token_ids = None
        response_token_ids = None

        # Path 1: direct attribute (openai 1.x / vLLM response object)
        if hasattr(return_value, "prompt_token_ids"):
            prompt_token_ids = return_value.prompt_token_ids
        if hasattr(return_value, "response_token_ids"):
            response_token_ids = return_value.response_token_ids

        # Path 2: model_extra dict (openai 2.x stores unknown vLLM fields here)
        if (prompt_token_ids is None
                and hasattr(return_value, "model_extra")
                and isinstance(return_value.model_extra, dict)):
            prompt_token_ids = return_value.model_extra.get("prompt_token_ids")
            response_token_ids = return_value.model_extra.get("response_token_ids")

        # Path 3: LiteLLM LegacyAPIResponse
        if prompt_token_ids is None and hasattr(return_value, "http_response") and hasattr(return_value.http_response, "json"):
            json_data = return_value.http_response.json()
            if isinstance(json_data, dict):
                prompt_token_ids = json_data.get("prompt_token_ids")
                response_token_ids = json_data.get("response_token_ids")

        if prompt_token_ids is not None:
            attributes["prompt_token_ids"] = list(prompt_token_ids)
        if response_token_ids is not None:
            # response_token_ids is [[token, ...]] — take the first (only) sequence
            if isinstance(response_token_ids, (list, tuple)) and len(response_token_ids) > 0:
                inner = response_token_ids[0]
                attributes["response_token_ids"] = list(inner) if not isinstance(inner, int) else list(response_token_ids)
            else:
                attributes["response_token_ids"] = list(response_token_ids)

        return attributes

    agentops.instrumentation.providers.openai.wrappers.chat.handle_chat_attributes = _handle_chat_attributes_with_tokens
    agentops.instrumentation.providers.openai.stream_wrapper.handle_chat_attributes = (
        _handle_chat_attributes_with_tokens
    )
    logger.info("Patched newer version of agentops using handle_chat_attributes")
    return True


def _unpatch_new_agentops():
    import agentops.instrumentation.providers.openai.wrappers.chat
    import agentops.instrumentation.providers.openai.stream_wrapper

    global _original_handle_chat_attributes
    if _original_handle_chat_attributes is not None:
        agentops.instrumentation.providers.openai.wrappers.chat.handle_chat_attributes = (
            _original_handle_chat_attributes
        )
        agentops.instrumentation.providers.openai.stream_wrapper.handle_chat_attributes = (
            _original_handle_chat_attributes
        )
        _original_handle_chat_attributes = None
        logger.info("Unpatched newer version of agentops using handle_chat_attributes")


def _patch_old_agentops():
    import opentelemetry.instrumentation.openai.shared.chat_wrappers
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import _handle_response, dont_throw

    global _original_handle_response
    _original_handle_response = _handle_response

    @dont_throw
    def _handle_response_with_tokens(response, span, *args, **kwargs):
        _original_handle_response(response, span, *args, **kwargs)
        if hasattr(response, "prompt_token_ids"):
            span.set_attribute("prompt_token_ids", list(response.prompt_token_ids))
        if hasattr(response, "response_token_ids"):
            span.set_attribute("response_token_ids", list(response.response_token_ids[0]))

        # For LiteLLM, response is a openai._legacy_response.LegacyAPIResponse
        if hasattr(response, "http_response") and hasattr(response.http_response, "json"):
            json_data = response.http_response.json()
            if isinstance(json_data, dict):
                if "prompt_token_ids" in json_data:
                    span.set_attribute("prompt_token_ids", list(json_data["prompt_token_ids"]))
                if "response_token_ids" in json_data:
                    span.set_attribute("response_token_ids", list(json_data["response_token_ids"][0]))

    opentelemetry.instrumentation.openai.shared.chat_wrappers._handle_response = _handle_response_with_tokens
    logger.info("Patched earlier version of agentops using _handle_response")
    return True


def _unpatch_old_agentops():
    import opentelemetry.instrumentation.openai.shared.chat_wrappers

    global _original_handle_response
    if _original_handle_response is not None:
        opentelemetry.instrumentation.openai.shared.chat_wrappers._handle_response = _original_handle_response
        _original_handle_response = None
        logger.info("Unpatched earlier version of agentops using _handle_response")


def _create_openai_compat_shims():
    """Create openai 2.x compatibility shims for agentops 0.4.x.

    agentops 0.4.x still references several openai 1.x internal modules that
    were removed in openai 2.x:
      - openai.resources.beta.chat  (removed; promoted to openai.resources.chat)
      - openai.resources._legacy_response  (removed internal module)

    We create thin shims / stubs so that agentops imports succeed without errors.
    """
    import sys
    import types

    # --- Shim 1: openai.resources.beta.chat ---
    if "openai.resources.beta.chat" not in sys.modules:
        try:
            import openai.resources.chat as _chat

            if "openai.resources.beta" not in sys.modules:
                import openai.resources as _res
                _beta = types.ModuleType("openai.resources.beta")
                sys.modules["openai.resources.beta"] = _beta
                setattr(_res, "beta", _beta)

            _beta = sys.modules["openai.resources.beta"]
            sys.modules["openai.resources.beta.chat"] = _chat
            setattr(_beta, "chat", _chat)
            logger.info("Created openai.resources.beta.chat shim for openai 2.x compatibility")
        except Exception as exc:
            logger.warning(f"Could not create openai.resources.beta.chat shim: {exc}")

    # --- Shim 2: openai.resources._legacy_response ---
    # agentops stream_wrapper imports this removed module; stub it out so the
    # import succeeds (the wrapper will fail gracefully when it actually tries
    # to use it, but that code path isn't exercised for non-streaming calls).
    if "openai.resources._legacy_response" not in sys.modules:
        try:
            import openai.resources as _res_mod
            _legacy = types.ModuleType("openai.resources._legacy_response")
            # Provide a no-op LegacyAPIResponse class so attribute lookups work
            class LegacyAPIResponse:
                pass
            _legacy.LegacyAPIResponse = LegacyAPIResponse
            sys.modules["openai.resources._legacy_response"] = _legacy
            setattr(_res_mod, "_legacy_response", _legacy)
            logger.info("Created openai.resources._legacy_response stub for openai 2.x compatibility")
        except Exception as exc:
            logger.warning(f"Could not create openai.resources._legacy_response stub: {exc}")


def instrument_agentops():
    """
    Instrument agentops to capture token IDs.
    Automatically detects and uses the appropriate patching method based on the installed agentops version.
    """
    # Create openai 2.x compatibility shims BEFORE agentops initialises its
    # OpenAI instrumentor, which still looks for openai 1.x internal modules.
    _create_openai_compat_shims()

    # Try newest version first (tested for 0.4.16)
    try:
        return _patch_new_agentops()
    except ImportError as e:
        logger.debug(f"Couldn't patch newer version of agentops: {str(e)}")

    # Note: 0.4.15 needs another patching method, but it's too shortlived to be worth handling separately.

    # Try older version (tested for 0.4.13)
    try:
        return _patch_old_agentops()
    except ImportError as e:
        logger.warning(f"Couldn't patch older version of agentops: {str(e)}")
        logger.error("Failed to instrument agentops - neither patching method was successful")
        return False


def uninstrument_agentops():
    try:
        _unpatch_new_agentops()
    except Exception:
        pass
    try:
        _unpatch_old_agentops()
    except Exception:
        pass


def agentops_local_server():
    """
    Returns a Flask app that can be used to test agentops integration.
    This server provides endpoints for token fetching and a catch-all endpoint.
    """
    app = flask.Flask(__name__)

    @app.route("/v3/auth/token", methods=["POST"])
    def fetch_token():
        return {"token": "dummy", "project_id": "dummy"}

    @app.route("/", defaults={"path": ""}, methods=["GET", "POST"])
    @app.route("/<path:path>", methods=["GET", "POST"])
    def catch_all(path):
        return {"path": path}

    return app


def _run_server(**kwargs):
    """
    Internal function to run the Flask server.
    This is used to avoid issues with multiprocessing and Flask's reloader.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore SIGINT in worker processes
    setproctitle.setproctitle(multiprocessing.current_process().name)
    app = agentops_local_server()
    app.run(**kwargs)


class AgentOpsServerManager:
    def __init__(self, daemon: bool = True, port: int | None = None):
        self.server_process: multiprocessing.Process | None = None
        self.server_port = port
        self.daemon = daemon
        logger.info("AgentOpsServerManager initialized.")

    def _find_available_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def start(self):
        if self.server_process and self.server_process.is_alive():
            logger.warning("AgentOps server process appears to be already running.")
            return

        if self.server_port is None:
            self.server_port = self._find_available_port()

        logger.info(f"Starting AgentOps local server on port {self.server_port}...")

        self.server_process = multiprocessing.Process(
            target=_run_server,
            kwargs={"host": "127.0.0.1", "port": self.server_port, "use_reloader": False, "debug": False},
            daemon=self.daemon,
            name="AgentFlow-AgentOpsServer",
        )
        self.server_process.start()
        logger.info(
            f"AgentOps local server process (PID: {self.server_process.pid}) started, targeting port {self.server_port}."
        )
        time.sleep(0.5)  # Brief wait for server to start up
        if not self.server_process.is_alive():
            logger.error(f"AgentOps local server failed to start or exited prematurely.")

    def is_alive(self) -> bool:
        if self.server_process and self.server_process.is_alive():
            return True
        return False

    def stop(self):
        if self.is_alive():
            logger.info(f"Stopping AgentOps local server (PID: {self.server_process.pid})...")
            self.server_process.terminate()  # Send SIGTERM
            self.server_process.join(timeout=5)  # Wait for clean exit
            if self.server_process.is_alive():
                logger.warning(
                    f"AgentOps server (PID: {self.server_process.pid}) did not terminate gracefully, killing..."
                )
                self.server_process.kill()  # Force kill
                self.server_process.join(timeout=10)  # Wait for kill
            self.server_process = None
            logger.info(f"AgentOps local server stopped.")
        else:
            logger.info("AgentOps local server was not running or already stopped.")

    def get_port(self) -> int | None:
        # Check liveness again in case it died since start()
        if self.is_alive() and self.server_port is not None:
            return self.server_port
        # If called after server stopped or failed, port might be stale or None
        if self.server_port is not None and (self.server_process is None or not self.server_process.is_alive()):
            logger.warning(
                f"AgentOps server port {self.server_port} is stored, but server process is not alive. Returning stored port."
            )
        return self.server_port
