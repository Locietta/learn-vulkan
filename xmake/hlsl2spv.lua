rule("hlsl2spv")
    set_extensions(".hlsl")
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
        local basename_with_type = path.basename(sourcefile_hlsl)
        local shadertype = string.sub(path.extension(basename_with_type), 2, -1)

        if shadertype == "" then 
            -- if not specify shader type, considered it a header, skip
            print("[WARN] hlsl2spv: shader type not specified, skip %s", sourcefile_hlsl)
            return
        end

        local targetenv = target:extraconf("rules", "hlsl2spv", "targetenv") or "vulkan1.0"
        local outputdir = target:extraconf("rules", "hlsl2spv", "outputdir") or path.join(target:autogendir(), "rules", "hlsl2spv")
        local hlslversion = target:extraconf("rules", "hlsl2spv", "hlslversion") or "2018"
        local spvfilepath = path.join(outputdir, basename_with_type .. ".spv")
        
        local shadermodel = target:extraconf("rules", "hlsl2spv", "shadermodel") or "6.0"
        local _sm = string.gsub(shadermodel, "%.", "_")

        local dxc_profile = shadertype .. "_" .. _sm

        batchcmds:show_progress(opt.progress, "${color.build.object}compiling.hlsl %s", sourcefile_hlsl)
        batchcmds:mkdir(outputdir)

        batchcmds:vrunv(dxc.program, {path(sourcefile_hlsl), "-spirv", "-HV", hlslversion, "-fspv-target-env=" .. targetenv, "-E", "main", "-T", dxc_profile, "-Fo", path(spvfilepath)})

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

    after_clean(function (target, batchcmds, sourcefile_hlsl) 
        import("private.action.clean.remove_files")

        local outputdir = target:extraconf("rules", "hlsl2spv", "outputdir") or path.join(target:targetdir(), "shader")
        remove_files(path.join(outputdir, "*.spv"))
        remove_files(path.join(outputdir, "*.spv.h"))
    end)