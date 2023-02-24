rule("hlsl2spv")
    set_extensions(".hlsl", ".vert", ".tesc", ".tese", ".geom", ".comp", ".frag", ".comp", ".mesh", ".ampl")
    on_load(function (target)
        local is_bin2c = target:extraconf("rules", "hlsl2spv", "bin2c")
        if is_bin2c then 
            local headerdir = path.join(target:autogendir(), "rules", "hlsl2spv")
            if not os.isdir(headerdir) then 
                os.mkdir(headerdir)
            end 
            target:add("includedirs", headerdir)
        end
    end)

    before_buildcmd_file(function (target, batchcmds, sourcefile_hlsl, opt) 
        import("lib.detect.find_tool")
        
        local dxc = find_tool("dxc")
        assert(dxc, "dxc not found!")

        -- hlsl to spv
        local extension = path.extension(sourcefile_hlsl)
        local shadertype = target:extraconf("rules", "hlsl2spv", "shadertype")

        assert(extension ~= ".hlsl" or shadertype ~= nil, "Shader type must be provided if it can't be inferred from file extensions!")

        local targetenv = target:extraconf("rules", "hlsl2spv", "targetenv") or "vulkan1.0"
        local outputdir = target:extraconf("rules", "hlsl2spv", "outputdir") or target:targetdir()
        local spvfilepath = path.join(outputdir, path.filename(sourcefile_hlsl) .. ".spv")
        
        local shadermodel = target:extraconf("rules", "hlsl2spv", "shadermodel") or "6.0"
        local _sm = string.gsub(shadermodel, "%.", "_")

        local shadertype_map = {
            [".vert"] = "vs",
            [".tesc"] = "hs",
            [".tese"] = "ds",
            [".geom"] = "gs",
            [".frag"] = "ps",
            [".comp"] = "cs",
            [".mesh"] = "ms",
            [".ampl"] = "as",
        }

        local _shadertype = shadertype or shadertype_map[extension]
        local dxc_profile = _shadertype .. "_" .. _sm

        batchcmds:show_progress(opt.progress, "${color.build.object}compiling.hlsl %s", sourcefile_hlsl)
        batchcmds:mkdir(outputdir)

        batchcmds:vrunv(dxc.program, {path(sourcefile_hlsl), "-spirv", "-fspv-target-env=" .. targetenv, "-E", "main", "-T", dxc_profile, "-Fo", path(spvfilepath)})

        -- bin2c
        local outputfile = spvfilepath
        local is_bin2c = target:extraconf("rules", "hlsl2spv", "bin2c")

        if is_bin2c then 
            -- get header file
            local headerdir = outputdir
            local headerfile = path.join(headerdir, path.filename(spvfilepath) .. ".h")

            target:add("includedirs", headerdir)
            outputfile = headerfile

            -- add commands
            local argv = {"lua", "private.utils.bin2c", "--nozeroend", "-i", path(spvfilepath), "-o", path(headerfile)}
            batchcmds:vrunv(os.programfile(), argv, {envs = {XMAKE_SKIP_HISTORY = "y"}})
        end 

        batchcmds:add_depfiles(sourcefile_hlsl)
        batchcmds:set_depmtime(os.mtime(outputfile))
        batchcmds:set_depcache(target:dependfile(outputfile))
    end)