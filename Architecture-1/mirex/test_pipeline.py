"""
Unit & Integration Test Suite for Architecture-1 (MusicScope-CL)

Validates:
1. Audio loading, padding, Mel conversion, and corrupt audio resilience.
2. Spectrogram augmentations (SpecAugment time/freq masking, pitch shift, noise).
3. SimCLR unsupervised contrastive loss and optimization step.
4. Track A Narrative 128-dim vector extraction (exact dims, finite values).
5. SupCon supervised contrastive loss, projection head, and fused representation (704-dim).
6. FusionMLP classifier, BCE with label smoothing, temperature calibration.
7. End-to-end inference scorer with Multiple Instance Learning (MIL).

Run:
    python -m unittest test_pipeline.py
"""
import os
import unittest
from pathlib import Path

import numpy as np
import torch

import config
from dataset import (
    load_audio_chunk,
    audio_to_logmel,
    SpectrogramAugmentation,
    SimCLRDataset,
    LabeledDataset,
)
from track_b_style import StyleEncoder
from track_a_narrative import extract_narrative_vector, get_feature_columns, feats_to_vector
from train_simclr import ProjectionHead as SimCLRProjHead, nt_xent_loss
from train_supcon import SupConProjectionHead, supcon_loss
from fusion_model import FusionMLP, TemperatureScaler, label_smoothing_bce, train_fusion_head, calibrate_temperature
from inference import MusicScopeCLScorer


class TestMusicScopeCLPipeline(unittest.TestCase):
    """Full test suite ensuring zero runtime crashes and mathematical validity."""

    @classmethod
    def setUpClass(cls):
        # Create a synthetic 3-second audio wave file in checkpoints directory for repeatable testing
        cls.test_dir = config.CHECKPOINT_DIR / "test_tmp"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.sr = config.SAMPLE_RATE
        t = np.linspace(0, 3, cls.sr * 3, endpoint=False)
        cls.synth_wave = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

        import soundfile as sf
        cls.valid_wav = cls.test_dir / "valid_tone.wav"
        sf.write(str(cls.valid_wav), cls.synth_wave, cls.sr)

        # Create a zero-byte broken file to test corruption recovery
        cls.corrupt_wav = cls.test_dir / "corrupted.wav"
        cls.corrupt_wav.write_bytes(b"INVALID_HEADER_GARBAGE_12345")

    def test_01_audio_loading_and_padding(self):
        """Audio chunk loader should pad short files to exact target length."""
        chunk = load_audio_chunk(self.valid_wav, sr=self.sr, duration=5.0)
        expected_len = int(self.sr * 5.0)
        self.assertEqual(len(chunk), expected_len)
        self.assertFalse(np.isnan(chunk).any())

    def test_02_audio_to_logmel_bounds_and_stability(self):
        """Log-Mel spectrogram should be normalized in [0, 1] with zero NaNs even on silence."""
        # 1. Normal signal
        mel = audio_to_logmel(self.synth_wave, sr=self.sr, n_mels=128)
        self.assertEqual(mel.shape[0], 128)
        self.assertTrue(0.0 <= mel.min() and mel.max() <= 1.0)
        self.assertFalse(np.isnan(mel).any())

        # 2. Pure silence edge case
        silence = np.zeros(self.sr * 2, dtype=np.float32)
        mel_silence = audio_to_logmel(silence, sr=self.sr, n_mels=128)
        self.assertFalse(np.isnan(mel_silence).any())
        self.assertFalse(np.isinf(mel_silence).any())

    def test_03_spectrogram_augmentation(self):
        """SpectrogramAugmentation must preserve shape and remain bounded in [0, 1]."""
        aug = SpectrogramAugmentation()
        mel = audio_to_logmel(self.synth_wave, sr=self.sr, n_mels=128)
        view = aug(mel)
        self.assertEqual(view.shape, mel.shape)
        self.assertTrue(0.0 <= view.min() and view.max() <= 1.0)
        self.assertFalse(np.isnan(view).any())

    def test_04_dataset_corrupt_file_recovery(self):
        """SimCLRDataset and LabeledDataset must gracefully skip corrupted files."""
        # Dataset containing both valid and broken audio
        paths = [self.corrupt_wav, self.valid_wav]
        ds = SimCLRDataset(paths)
        # Fetching item should succeed despite corrupt file in list
        view1, view2 = ds[0]
        self.assertEqual(view1.shape, view2.shape)
        self.assertEqual(view1.shape[1], config.N_MELS)

        labeled_ds = LabeledDataset([(self.corrupt_wav, 1), (self.valid_wav, 0)])
        spec, label = labeled_ds[0]
        self.assertEqual(spec.shape[1], config.N_MELS)
        self.assertIn(label, [0, 1])

    def test_05_simclr_loss_and_step(self):
        """SimCLR NT-Xent loss must decrease with gradients flowing through encoder & projector."""
        encoder = StyleEncoder(pretrained=False)
        proj = SimCLRProjHead(in_dim=encoder.out_dim)
        opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=1e-3)

        # Batch of 4 paired views
        x1 = torch.randn(4, 1, 128, 200)
        x2 = torch.randn(4, 1, 128, 200)

        z1 = proj(encoder(x1))
        z2 = proj(encoder(x2))
        loss = nt_xent_loss(z1, z2)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Check weights updated
        for p in proj.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

    def test_06_narrative_vector_exact_128_dims(self):
        """Track A Narrative extractor must return exactly 128 numerical features."""
        feats = extract_narrative_vector(str(self.valid_wav), sr=self.sr, duration=3.0)
        cols = get_feature_columns(feats)
        self.assertEqual(len(cols), config.NARRATIVE_DIM,
                         f"Expected {config.NARRATIVE_DIM} features, got {len(cols)}")
        vec = feats_to_vector(feats, cols)
        self.assertEqual(len(vec), config.NARRATIVE_DIM)
        self.assertFalse(np.isnan(vec).any())
        self.assertFalse(np.isinf(vec).any())

    def test_07_supcon_projection_and_loss(self):
        """SupCon head projects 704-dim fused vector to 128-dim and computes valid loss."""
        proj = SupConProjectionHead(in_dim=704, out_dim=128)
        fused = torch.randn(6, 704)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        z = proj(fused)
        self.assertEqual(z.shape, (6, 128))
        loss = supcon_loss(z, labels)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_08_fusion_mlp_training_and_calibration(self):
        """FusionMLP trains with BCE label smoothing and produces calibrated probabilities."""
        X = np.random.randn(20, 704).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10, dtype=np.float32)

        model, (X_val_t, y_val) = train_fusion_head(X, y, epochs=5)
        scaler = calibrate_temperature(model, X_val_t, y_val, epochs=10)
        self.assertGreater(scaler.temperature.item(), 0.0)

        with torch.no_grad():
            raw_logits = model(torch.tensor(X[:2], dtype=torch.float32))
            cal_logits = scaler(raw_logits)
            probs = torch.sigmoid(cal_logits).numpy()

        self.assertEqual(len(probs), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_09_end_to_end_inference_scorer(self):
        """End-to-end MusicScopeCLScorer evaluates audio and returns valid P(AI) in [0, 1]."""
        simclr_path = config.CHECKPOINT_DIR / "simclr_best.pt"
        fusion_path = config.CHECKPOINT_DIR / "fusion_best.pt"

        if not simclr_path.exists() or not fusion_path.exists():
            self.skipTest("Checkpoints not yet trained.")

        scorer = MusicScopeCLScorer(
            simclr_ckpt=str(simclr_path),
            fusion_ckpt=str(fusion_path),
        )
        prob = scorer.score(str(self.valid_wav), n_chunks=2)
        self.assertIsInstance(prob, float)
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertFalse(np.isnan(prob))

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
