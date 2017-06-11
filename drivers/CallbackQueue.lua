require 'torch'
local _ = require 'moses'
local hdf5 = require 'hdf5'

local CallbackQueue = torch.class('CallbackQueue')
-- supports the training event loop
-- this would be best implemented as a heap, but even this is [massive] overkill

function CallbackQueue:__init(startIter)
  self.q = {}
  self.iter = startIter or 1
  self.oneshots = 0
  -- one-indexed because lua
end

function CallbackQueue:add(cb)
  assert(type(cb.cb) == 'function')

  cb.priority = cb.priority or -math.huge

  if cb.iter and cb.iter < self.iter then
    return
  elseif cb.iter then
    -- insert at correct position
    local sortByIter = function(cb1, cb2)
      return cb1.iter == cb2.iter and cb1.priority < cb2.priority or cb1.iter < cb2.iter
    end
    local insertInd = _.sortedIndex(self.q, cb, sortByIter)
    table.insert(self.q, insertInd, cb)

    if cb.interval == nil then
      self.oneshots = self.oneshots + 1
    end
  elseif cb.interval then
    cb.iter = 1
    table.insert(self.q, 1, cb)
  end
end

function CallbackQueue:waitTime()
  return #self.q == 0 and math.huge or math.max(self.q[1].iter - self.iter + 1, 0)
end

function CallbackQueue:advance(nIters)
  nIters = nIters or (#self.q > 0 and self:waitTime() or 0)
  self.iter = self.iter + nIters
end

function CallbackQueue:pull()
  -- returns an iterator over all events that should happen on or before self.iter
  return function()
    if self.q[1] and self.q[1].iter < self.iter then
      local event = table.remove(self.q, 1)
      if event.interval then
        local reiter = _.template({
          cb=event.cb,
          iter=self.iter+event.interval-1,
          interval=event.interval
        }, event)
        self:add(reiter)
      else
        self.oneshots = self.oneshots - 1
      end
      return event.cb
    end
  end
end

function CallbackQueue:__len() return self.oneshots end
