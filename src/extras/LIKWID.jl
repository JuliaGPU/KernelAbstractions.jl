module LIKWID
  module Marker
    using Libdl
    const liblikwid = Libdl.find_library("liblikwid")
    function init()
       ccall((:likwid_markerInit, liblikwid),
	     Cvoid, ())
    end

    function threadinit()
       ccall((:likwid_markerThreadInit, liblikwid),
	     Cvoid, ())
    end

    function registerregion(regiontag)
       ccall((:likwid_markerRegisterRegion, liblikwid),
	     Cint, (Cstring,), regiontag)
    end

    function startregion(regiontag)
       ccall((:likwid_markerStartRegion, liblikwid),
	     Cint, (Cstring,), regiontag)
    end
    function stopregion(regiontag)
       ccall((:likwid_markerStopRegion, liblikwid),
	     Cint, (Cstring,), regiontag)
    end
    # markerGetRegion
    # markerNextGroup
    function resetregion(regiontag)
       ccall((:likwid_markerResetRegion, liblikwid),
	     Cint, (Cstring,), regiontag)
    end
    function close()
       ccall((:likwid_markerClose, liblikwid),
	     Cvoid, ())
    end
  end
end
