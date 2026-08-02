"""v0.7: minimal OpenTelemetry tracing.

Scope note: exports spans to the console, not a real backend (Jaeger/Tempo/
an OTLP collector). Standing up a trace-storage service is out of scope for
what this milestone is actually verifying -- that spans are created with
the right structure and attributes around the real request path. Swapping
the exporter for an OTLP one pointed at a real collector is a one-line
change (`OTLPSpanExporter` instead of `ConsoleSpanExporter`); the
instrumentation itself doesn't change.
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_provider = TracerProvider()
_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("metal-inference-server")
