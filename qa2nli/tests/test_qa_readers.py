import pytest
from qa2nli.qa_readers.race import RaceReader, OriginalRaceSample, ANS_LETTER_TO_NUM
from qa2nli.qa_readers.reader import SingleQuestionSample, ConversionManager, PureNLISample
from qa2nli.qa_readers.writer import JSONWriter
from pathlib import Path
from dataclasses import asdict
from .converters import dummy_converter, basic_bart_converter, bart_with_spacy_converter
from .data import (race_dev_data, race_dev_data_as_dataclasses,
                   race_dev_data_as_dataclasses_raw, race_data_path)
import json
import itertools
import tqdm
import logging
logger = logging.getLogger(__file__)


def test_race_reader(race_dev_data, race_data_path):
    original_data = race_dev_data
    race_data = RaceReader(
        input_type='OriginalRaceSample',
        output_type='SingleQuestionSample').read(race_data_path)

    for sample in race_data:
        file_id, q_num = (sample.id).split('_')
        q_num = int(q_num)
        target_file = original_data[file_id]
        target_sample = SingleQuestionSample(
            id=target_file['id'] + f'_{q_num}',
            article=target_file['article'],
            question=target_file['questions'][q_num],
            options=target_file['options'][q_num],
            answer=ANS_LETTER_TO_NUM[target_file['answers'][q_num]])
        assert target_sample == sample


# def test_conversion_manager(dummy_converter, race_data_path):
#    race_data = RaceReader(
#        input_type='OriginalRaceSample',
#        output_type='SingleQuestionSample').read(race_data_path)
#    manager = ConversionManager(race_data, batch_size=2)
#
#    for batch in manager:
#        converted = dummy_converter(*batch)
#        manager.tell(converted)
#    converted = [c for c in manager.converted()]
#    assert len(converted) == len(race_data)
#
#    for cov, ori in zip(converted, race_data):
#        assert cov.id == ori.id
#        assert cov.hypothesis_options == [(ori.question + ' ' + s)
#                                          for s in ori.options]
#
#
# def test_conversion_manager_tqdm(dummy_converter, race_data_path):
#    race_data = RaceReader(
#        input_type='OriginalRaceSample',
#        output_type='SingleQuestionSample').read(race_data_path)
#    manager = ConversionManager(race_data, batch_size=2)
#
#    for batch in tqdm.tqdm(manager):
#        converted = dummy_converter(*batch)
#        manager.tell(converted)


def test_SingleQuestionSample_to_NLIWithOptions(dummy_converter,
                                                race_data_path):
    SingleQuestionSample.to_nli_converter = dummy_converter
    race_data = RaceReader(
        input_type='OriginalRaceSample',
        output_type='SingleQuestionSample').read(race_data_path)
    converted = [sample.to('NLIWithOptionsSample')[0] for sample in race_data]

    for cov, ori in zip(converted, race_data):
        assert cov.id == ori.id
        assert cov.hypothesis_options == [(ori.question + ' ' + s)
                                          for s in ori.options]


def test_SingleQuestionSample_to_PureNLI(dummy_converter, race_data_path):
    SingleQuestionSample.to_nli_converter = dummy_converter
    race_data = RaceReader(
        input_type='OriginalRaceSample',
        output_type='SingleQuestionSample').read(race_data_path)
    converted = [sample.to('PureNLISample') for sample in race_data]

    for cov, ori in zip(converted, race_data):
        for i, single_nli in enumerate(cov):
            single_nli.id == ori.id + f"_{i}"
            assert single_nli.hypothesis == (
                ori.question + ' ' + ori.options[i])


def test_SingleQuestionSample_to_PureNLI_bart(basic_bart_converter,
                                              race_data_path):
    SingleQuestionSample.to_nli_converter = basic_bart_converter
    race_data = RaceReader(
        input_type='OriginalRaceSample',
        output_type='SingleQuestionSample').read(race_data_path)
    converted = [sample.to('PureNLISample') for sample in race_data]

    for i, converted_group in enumerate(converted):
        if i >= 2:
            break

        for j, ex in enumerate(converted_group):
            if j >= 2:
                break
            logger.info(ex)


def test_SingleQuestionSample_to_PureNLI_bart_with_prepostprocessors(
        bart_with_spacy_converter, race_data_path):
    SingleQuestionSample.to_nli_converter = bart_with_spacy_converter
    race_data = RaceReader(
        input_type='OriginalRaceSample',
        output_type='SingleQuestionSample').read(race_data_path)
    converted = [sample.to('PureNLISample') for sample in race_data]

    for i, converted_group in enumerate(converted):
        if i >= 2:
            break

        for j, ex in enumerate(converted_group):
            if j >= 2:
                break
            logger.info(ex)


def test_SingleQuestionSample_to_NLIWithOptions_bart_with_prepostprocessors(
        bart_with_spacy_converter, race_data_path):
    SingleQuestionSample.to_nli_converter = bart_with_spacy_converter
    race_data = RaceReader(
        input_type='OriginalRaceSample',
        output_type='SingleQuestionSample').read(race_data_path)
    converted = [sample.to('NLIWithOptionsSample')[0] for sample in race_data]

    for i, ex in enumerate(converted):
        logger.info(f"Converted {i}: {ex}")

        if i > 4:
            break


def test_writer(race_dev_data_as_dataclasses_raw, tmpdir):
    file_ = tmpdir.join("race_dev.json")
    writer = JSONWriter(race_dev_data_as_dataclasses_raw)
    writer.write(file_)
    # read it back
    with open(file_) as f:
        written = [OriginalRaceSample(**s) for s in json.load(f)]

    assert written == race_dev_data_as_dataclasses_raw
