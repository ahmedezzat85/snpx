import numpy as np

try:
    from mvnc import mvncapi as mvnc
    class MvNCS(object):
        """ A simple wrapper for the Movidius NCS device.
        """
        def __init__(self, dev_idx=0, dont_block=False):
            # configure the NCS
            self.non_blocking = dont_block
            mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 1)
            devices = mvnc.EnumerateDevices()
            if len(devices) == 0: raise ValueError('No devices found')
            self.dev = mvnc.Device(devices[dev_idx])

            # Open the NCS
            try:
                self.dev.OpenDevice()
            except:
                raise ValueError('Cannot Open NCS Device')

        def load_model(self, model_file):
            with open(model_file, mode='rb') as f:
                blob = f.read()

            self.ncs_graph = self.dev.AllocateGraph(blob)
            self.ncs_graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
            if self.non_blocking is True:
                self.ncs_graph.SetGraphOption(mvnc.GraphOption.DONT_BLOCK, 1)

        def load_input(self, input_tensor):
            input_tensor = input_tensor.astype(np.float16)
            self.ncs_graph.LoadTensor(input_tensor, None)

        def get_output(self):
            ncs_out = None
            while ncs_out is None:
                ncs_out, _ = self.ncs_graph.GetResult()
            ncs_time = np.sum(self.ncs_graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN))
            return ncs_out, ncs_time

        def unload_model(self):
            self.ncs_graph.DeallocateGraph()

        def close(self):
            self.dev.CloseDevice()

except ImportError:
    class MvNCS(object):
        """ A simple wrapper for the Movidius NCS device.
        """
        def __init__(self, dev_idx=0, dont_block=False):
            raise NotImplementedError()
