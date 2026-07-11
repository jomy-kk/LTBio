# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: biosignals
# Module: Speech
# Description: Class Speech, a type of Biosignal, representative of the mechanical process produced by the vocal system
# and recorded as audio by a microphone.

# Contributors: Seyedali Divbandroudbaraki, João Saraiva
# Created: 13/05/2026
# Last Updated: 19/05/2026

# ===================================

from datetime import timedelta
from warnings import warn

import numpy as np
from numpy import average, array

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Timeseries import Timeseries
from ltbio.biosignals.timeseries.Unit import PulseCodeModulation


class Speech(Biosignal):
    DEFAULT_UNIT = PulseCodeModulation()

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(Speech, self).__init__(timeseries, source, patient, acquisition_location, name)

    @staticmethod
    def __silence_ratio(samples: np.ndarray, sampling_frequency: float,
        frame_duration: float = 0.02, threshold: float = 0.01) -> float:
        frame_length = int(sampling_frequency * frame_duration)
        if frame_length == 0 or len(samples) < frame_length:
            return 0.0
        n_frames = len(samples) // frame_length
        frames = samples[:n_frames * frame_length].reshape(n_frames, frame_length)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        return float(np.sum(rms < threshold) / n_frames)
    
    @staticmethod
    def __is_good_quality(samples: np.ndarray, sampling_frequency: float,
                          silence_threshold: float = 0.01,
                          max_silence_ratio: float = 0.8,
                          clipping_threshold: float = 0.99) -> bool:
        frame_length = max(1, int(sampling_frequency * 0.02))
        n_frames = len(samples) // frame_length
        if n_frames == 0:
            return False
        frames = samples[:n_frames * frame_length].reshape(n_frames, frame_length)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        if np.sum(rms < silence_threshold) / n_frames > max_silence_ratio:
                  return False
        if np.max(np.abs(samples)) >= clipping_threshold:
            return False
        return True

    def silence_percentage(self, by_segment: bool = False, threshold: float = 0.01) -> dict:
        """
        Computes the % of silence of each channel.
        If `by_segment` is True, a list of values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.

        :return: A dictionary of % of silence for each channel.
        """
        
        res = {}
        for channel_name, channel in self:
            values = channel._apply_operation_and_return(
                self.__silence_ratio,
                sampling_frequency = channel.sampling_frequency,
                threshold=threshold
            )
            if by_segment:
                res[channel_name] = values
            else:
                res[channel_name] = average(
                    array(values),
                    weights = [subdomain.timedelta.total_seconds() for subdomain in channel.domain]
                )
        return res

    def acceptable_quality(self):
        """
        Returns the periods of time, when the speech signal is not mostly silent and no clipping.
        """

        return self.when(
            lambda samples: self.__is_good_quality(samples, self.sampling_frequency),
            window=timedelta(seconds=10)
        )


    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
        # Future work

    def _decorate_plot(self, fig, axes, timeseries_plotting_method, title, xlabel, ylabel, show: bool, save_to: str = None):
        if timeseries_plotting_method is not Timeseries._plot or not show or save_to is not None:
            return
        try:
            audio, sampling_frequency, playback_to_plot_x = self._plot_audio_matrix()
            import sounddevice
        except (ImportError, ValueError) as error:
            warn(f"Speech playback controls were not added: {error}")
            return

        fig._ltbio_speech_playback_controller = _SpeechPlotPlaybackController(
            fig, axes, audio, sampling_frequency, sounddevice, playback_to_plot_x
        )

    def _plot_audio_matrix(self):
        channels = [channel for _, channel in self]
        if len(channels) == 0:
            raise ValueError("there are no channels to play")

        sampling_frequency = channels[0].sampling_frequency
        domain = channels[0].domain
        for channel in channels:
            if channel.sampling_frequency != sampling_frequency:
                raise ValueError("all Speech channels must have the same sampling frequency")
            if channel.domain != domain:
                raise ValueError("all Speech channels must have the same domain")

        audio_segments = []
        segment_lengths = []
        for segment_i in range(channels[0].n_segments):
            segment_samples = []
            segment_length = None
            for channel in channels:
                channel_samples = np.asarray(channel.segments[segment_i].samples)
                if channel_samples.ndim != 1:
                    raise ValueError("each Speech channel segment must contain one-dimensional samples")
                if segment_length is None:
                    segment_length = len(channel_samples)
                elif len(channel_samples) != segment_length:
                    raise ValueError("matching Speech channel segments must have the same number of samples")
                segment_samples.append(channel_samples)
            if segment_length == 0:
                raise ValueError("Speech playback does not support empty segments")
            segment_lengths.append(segment_length)
            audio_segments.append(np.column_stack(segment_samples))

        if len(audio_segments) == 0:
            raise ValueError("there are no samples to play")

        audio = np.vstack(audio_segments)
        playback_to_plot_x = self.__plot_x_from_segment_lengths(segment_lengths, sampling_frequency)
        return self.__normalize_audio_for_plot_playback(audio), sampling_frequency, playback_to_plot_x

    @staticmethod
    def __plot_x_from_segment_lengths(segment_lengths, sampling_frequency):
        space = int(sampling_frequency) * 2
        plot_x = []
        previous_last_x = None
        for segment_length in segment_lengths:
            segment_x = np.arange(segment_length)
            if previous_last_x is not None:
                segment_x = segment_x + previous_last_x + space
            plot_x.append(segment_x)
            previous_last_x = segment_x[-1]
        return np.concatenate(plot_x)

    @staticmethod
    def __normalize_audio_for_plot_playback(audio):
        if np.issubdtype(audio.dtype, np.integer):
            dtype_info = np.iinfo(audio.dtype)
            scale = max(abs(dtype_info.min), dtype_info.max)
            return audio.astype(np.float32) / scale

        audio = np.nan_to_num(audio.astype(np.float32), copy=False)
        peak = np.max(np.abs(audio))
        if peak > 1:
            audio = audio / peak
        return audio


class _SpeechPlotPlaybackController:

    CURSOR_UPDATE_INTERVAL_MS = 1000
    AUDIO_BLOCKSIZE = 8192

    def __init__(self, fig, axes, audio, sampling_frequency, sounddevice, playback_to_plot_x):
        from matplotlib.widgets import Button

        self.__fig = fig
        self.__axes = tuple(axes)
        self.__audio = np.ascontiguousarray(audio, dtype=np.float32)
        self.__sampling_frequency = sampling_frequency
        self.__sounddevice = sounddevice
        self.__stream = None
        self.__playback_to_plot_x = np.asarray(playback_to_plot_x)
        self.__current_sample = 0
        self.__is_playing = False
        self.__dragging = False

        self.__cursor_lines = [ax.axvline(0, color='black', linewidth=0.8, alpha=0.8) for ax in self.__axes]
        self.__button_axes = fig.add_axes([0.90, 0.94, 0.08, 0.04])
        self.__button = Button(self.__button_axes, "Play")
        self.__button.on_clicked(self.__toggle_playback)

        self.__timer = fig.canvas.new_timer(interval=self.CURSOR_UPDATE_INTERVAL_MS)
        self.__timer.add_callback(self.__on_timer)
        self.__press_event = fig.canvas.mpl_connect('button_press_event', self.__on_button_press)
        self.__motion_event = fig.canvas.mpl_connect('motion_notify_event', self.__on_mouse_motion)
        self.__release_event = fig.canvas.mpl_connect('button_release_event', self.__on_button_release)

    @property
    def current_sample(self):
        return self.__current_sample

    @property
    def is_playing(self):
        return self.__is_playing

    def __toggle_playback(self, _):
        if self.__is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if self.__current_sample >= len(self.__audio) - 1:
            self.__set_current_sample(0)
        try:
            self.__stream = self.__sounddevice.OutputStream(
                samplerate=self.__sampling_frequency,
                channels=self.__audio.shape[1],
                dtype='float32',
                latency='high',
                blocksize=self.AUDIO_BLOCKSIZE,
                callback=self.__audio_callback,
            )
            self.__stream.start()
        except Exception as error:
            warn(f"Could not start Speech playback: {error}")
            self.__close_stream()
            return

        self.__is_playing = True
        self.__timer.start()
        self.__set_button_label("Pause")

    def pause(self):
        self.__sync_current_sample()
        self.__close_stream()
        self.__is_playing = False
        self.__timer.stop()
        self.__set_button_label("Play")
        self.__draw_cursor()

    def seek(self, sample):
        self.__seek_to_plot_x(sample)

    def __finish_playback(self):
        self.__close_stream()
        self.__is_playing = False
        self.__timer.stop()
        self.__current_sample = len(self.__audio) - 1
        self.__set_button_label("Play")
        self.__draw_cursor()

    def __sync_current_sample(self):
        if self.__is_playing:
            self.__current_sample = min(self.__current_sample, len(self.__audio))

    def __audio_callback(self, outdata, frames, _, __):
        start = self.__current_sample
        end = min(start + frames, len(self.__audio))
        chunk = self.__audio[start:end]
        outdata[:len(chunk)] = chunk
        if len(chunk) < frames:
            outdata[len(chunk):].fill(0)
            self.__current_sample = len(self.__audio)
            callback_stop = getattr(self.__sounddevice, 'CallbackStop', None)
            if callback_stop is not None:
                raise callback_stop()
        else:
            self.__current_sample = end

    def __close_stream(self):
        if self.__stream is not None:
            self.__stream.stop()
            self.__stream.close()
            self.__stream = None

    def __on_timer(self):
        self.__sync_current_sample()
        if self.__current_sample >= len(self.__audio):
            self.__finish_playback()
        else:
            self.__draw_cursor()

    def __on_button_press(self, event):
        if self.__is_left_mouse_button(event.button) and event.inaxes in self.__axes and event.xdata is not None:
            self.__dragging = True
            self.__seek_to_plot_x(event.xdata)

    def __on_mouse_motion(self, event):
        if self.__dragging and event.inaxes in self.__axes and event.xdata is not None:
            self.__seek_to_plot_x(event.xdata)

    def __on_button_release(self, _):
        self.__dragging = False

    @staticmethod
    def __is_left_mouse_button(button):
        return button == 1 or getattr(button, "name", None) == "LEFT"

    def __seek_to_plot_x(self, plot_x):
        was_playing = self.__is_playing
        if was_playing:
            self.__close_stream()
            self.__timer.stop()
            self.__is_playing = False

        self.__set_current_sample(self.__sample_at_plot_x(plot_x))
        if was_playing:
            self.play()

    def __sample_at_plot_x(self, plot_x):
        insertion_point = np.searchsorted(self.__playback_to_plot_x, plot_x)
        if insertion_point <= 0:
            return 0
        if insertion_point >= len(self.__playback_to_plot_x):
            return len(self.__playback_to_plot_x) - 1

        previous_sample = insertion_point - 1
        next_sample = insertion_point
        if abs(self.__playback_to_plot_x[previous_sample] - plot_x) <= abs(self.__playback_to_plot_x[next_sample] - plot_x):
            return previous_sample
        return next_sample

    def __set_current_sample(self, sample):
        self.__current_sample = max(0, min(int(round(sample)), len(self.__audio) - 1))
        self.__draw_cursor()

    def __draw_cursor(self):
        x = self.__playback_to_plot_x[min(self.__current_sample, len(self.__audio) - 1)]
        for cursor_line in self.__cursor_lines:
            cursor_line.set_xdata((x, x))
        self.__fig.canvas.draw_idle()

    def __set_button_label(self, label):
        self.__button.label.set_text(label)
        self.__fig.canvas.draw_idle()
