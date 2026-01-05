#!/usr/bin/env python3
"""Совместимый entrypoint для обучения RQ-VAE.

Исторически файл был монолитным. Теперь реализация вынесена в пакет `mipt_master/src/rqvae`,
а этот файл оставлен как тонкий враппер для привычного запуска.
"""

from rqvae.cli import main


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    # Совместимый entrypoint: теперь основная реализация лежит в src/rqvae/*
    from mipt_master.src.rqvae.cli import main

    main()

