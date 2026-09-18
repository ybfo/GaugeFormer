# GaugeFormer

Code for **GaugeFormer: Historical Validation for Cross-System Forecast Adaptation**.

GaugeFormer uses completed forecasting tasks inside the observed context to select a local trajectory, decide whether it should contribute, and apply a controlled residual correction to a frozen backbone forecast. The requested future is used only for evaluation.

This repository contains the implementation and reproduction assets for the main manuscript. The same decision rule is evaluated around Moirai, Timer-XL and GTM, alongside UniTime, CPiRi, TEFN and TGGC.

The repository is private during preparation for submission.
