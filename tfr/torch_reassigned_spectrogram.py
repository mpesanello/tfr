import inspect
import torch
import torchaudio
import numpy as np
from examples.multicomponent_spectrograms import generate_example_sound

def tailored_args(f, kwargs):
    f_signature = inspect.signature(f)
    expected_params = [param.name for param in f_signature.parameters.values() if param.name != 'self']
    return {k: v for k, v in kwargs.items() if k in expected_params}


def db_scale(magnitude_spectrum):
    scaled = 20 * torch.log10(torch.maximum(torch.tensor(1e-6), magnitude_spectrum))
    return scaled


def arg(values):
    return (torch.angle(values) / (2 * np.pi)) % 1.0


class ReassignedSpectrogram(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        spectrogram_kwargs = tailored_args(torchaudio.transforms.Spectrogram.__init__, kwargs)
        self.power = spectrogram_kwargs.get("power", 1)
        spectrogram_kwargs["power"] = None
        if "n_fft" not in spectrogram_kwargs:
            spectrogram_kwargs["n_fft"] = spectrogram_kwargs.get("win_length", 400)

        self.spectrogram = torchaudio.transforms.Spectrogram(**spectrogram_kwargs)

        self.spectrogram.window = kwargs.get("window", self.spectrogram.window)

        reassignment_kwargs = {key: val for key, val in kwargs.items() if key not in spectrogram_kwargs}
        self.sample_rate = reassignment_kwargs.get("sample_rate", 44100)
        self.db_scale = reassignment_kwargs.get("db_scale", True)

    def get_inst_freqs(self, signal, spec_complex):
        time_shifted_signal = torch.roll(signal, shifts=1, dims=0)
        time_shifted_signal[0] = 0.0
        time_shifted_spec_complex = self.spectrogram(time_shifted_signal)
        cross_spectrum = spec_complex * time_shifted_spec_complex.conj()
        return arg(cross_spectrum)

    def get_inst_time_delays(self, spec_complex):
        freq_shifted_spec = torch.roll(spec_complex, shifts=1, dims=0)
        freq_shifted_spec[0, :] = 0.0
        cross_spectrum = spec_complex * freq_shifted_spec.conj()
        return 0.5 - arg(cross_spectrum)

    def forward(self, signal):
        if not torch.is_tensor(signal):
            signal = torch.tensor(signal)

        spec_complex = self.spectrogram(signal)
        spec_mag = torch.abs(spec_complex) / spec_complex.shape[0]

        inst_freqs = self.get_inst_freqs(signal, spec_complex)
        time_delays = self.get_inst_time_delays(spec_complex)

        win_length = self.spectrogram.win_length
        win_duration = win_length / self.sample_rate
        input_bin_count = inst_freqs.shape[0]

        duration = len(signal) / self.sample_rate
        win_start_times = torch.arange(0, duration, self.spectrogram.hop_length / self.sample_rate)

        eps = np.finfo(np.float32).eps
        win_center_times = torch.tile(win_start_times + (win_duration / 2) + eps, dims=(input_bin_count, 1))

        reassigned_times = win_center_times + time_delays * win_duration
        reassigned_freqs = inst_freqs
        # return reassigned_times, reassigned_freqs
        histogram_input = torch.stack((reassigned_freqs, reassigned_times), dim=2)

        output_frame_count = int(np.ceil((duration * self.sample_rate) / self.spectrogram.hop_length))
        time_range = (0, output_frame_count * self.spectrogram.hop_length / self.sample_rate)
        bin_range = (0.0, 0.5) if self.spectrogram.onesided else (0.0, 1.0)
        histogram_output = torch.histogramdd(input=histogram_input,
                                             range=[bin_range[0], bin_range[1], time_range[0], time_range[1]],
                                             weight=spec_mag,
                                             bins=spec_mag.shape)
        reassigned_spectrogram = histogram_output.hist ** self.power
        if self.db_scale:
            return db_scale(reassigned_spectrogram)
        return reassigned_spectrogram

