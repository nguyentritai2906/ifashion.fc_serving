# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/tensorflow_serving/apis/model.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import wrappers_pb2 as google_dot_protobuf_dot_wrappers__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*protos/tensorflow_serving/apis/model.proto\x12\x12tensorflow.serving\x1a\x1egoogle/protobuf/wrappers.proto\"\x8c\x01\n\tModelSpec\x12\x0c\n\x04name\x18\x01 \x01(\t\x12.\n\x07version\x18\x02 \x01(\x0b\x32\x1b.google.protobuf.Int64ValueH\x00\x12\x17\n\rversion_label\x18\x04 \x01(\tH\x00\x12\x16\n\x0esignature_name\x18\x03 \x01(\tB\x10\n\x0eversion_choiceB\x03\xf8\x01\x01\x62\x06proto3')



_MODELSPEC = DESCRIPTOR.message_types_by_name['ModelSpec']
ModelSpec = _reflection.GeneratedProtocolMessageType('ModelSpec', (_message.Message,), {
  'DESCRIPTOR' : _MODELSPEC,
  '__module__' : 'protos.tensorflow_serving.apis.model_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ModelSpec)
  })
_sym_db.RegisterMessage(ModelSpec)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _MODELSPEC._serialized_start=99
  _MODELSPEC._serialized_end=239
# @@protoc_insertion_point(module_scope)
