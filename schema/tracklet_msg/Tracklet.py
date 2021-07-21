# automatically generated by the FlatBuffers compiler, do not modify

# namespace: 

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Tracklet(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Tracklet()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTracklet(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Tracklet
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Tracklet
    def TrackId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    # Tracklet
    def Bbox(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = o + self._tab.Pos
            from .Bbox import Bbox
            obj = Bbox()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Tracklet
    def Classid(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    # Tracklet
    def Score(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # Tracklet
    def Classscore(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

def Start(builder): builder.StartObject(5)
def TrackletStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddTrackId(builder, trackId): builder.PrependUint64Slot(0, trackId, 0)
def TrackletAddTrackId(builder, trackId):
    """This method is deprecated. Please switch to AddTrackId."""
    return AddTrackId(builder, trackId)
def AddBbox(builder, bbox): builder.PrependStructSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(bbox), 0)
def TrackletAddBbox(builder, bbox):
    """This method is deprecated. Please switch to AddBbox."""
    return AddBbox(builder, bbox)
def AddClassid(builder, classid): builder.PrependUint16Slot(2, classid, 0)
def TrackletAddClassid(builder, classid):
    """This method is deprecated. Please switch to AddClassid."""
    return AddClassid(builder, classid)
def AddScore(builder, score): builder.PrependFloat32Slot(3, score, 0.0)
def TrackletAddScore(builder, score):
    """This method is deprecated. Please switch to AddScore."""
    return AddScore(builder, score)
def AddClassscore(builder, classscore): builder.PrependFloat32Slot(4, classscore, 0.0)
def TrackletAddClassscore(builder, classscore):
    """This method is deprecated. Please switch to AddClassscore."""
    return AddClassscore(builder, classscore)
def End(builder): return builder.EndObject()
def TrackletEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)