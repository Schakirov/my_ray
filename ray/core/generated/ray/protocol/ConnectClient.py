# automatically generated by the FlatBuffers compiler, do not modify

# namespace: protocol

import flatbuffers

class ConnectClient(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsConnectClient(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConnectClient()
        x.Init(buf, n + offset)
        return x

    # ConnectClient
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConnectClient
    def ClientId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def ConnectClientStart(builder): builder.StartObject(1)
def ConnectClientAddClientId(builder, clientId): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(clientId), 0)
def ConnectClientEnd(builder): return builder.EndObject()
