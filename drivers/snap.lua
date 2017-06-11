require 'dpnn'
require 'nngraph'

return function(model, workers, opts, state)
  local OUTFILE_TMP = opts.snapfile..'_i%s_v%.3f.dat'

  bestPerf = -math.huge
  return function()

    if state.valPerf == nil or state.valPerf > bestPerf then
      bestPerf = state.valPerf or bestPerf
      local outfile = string.format(OUTFILE_TMP, state.t, bestPerf)
      print('Saving model to '..outfile..'...')

      if opts.ngpus>1 then
        paths.dofile('../drivers/utils.lua')
        saveParallel(outfile,model,opts)
      else
        torch.save(outfile,model:clearState())
      end

      print("Saved")
      state.valtrack = 0
    else
      state.valtrack = state.valtrack + 1
    end
  end
end
