import string

import pandas as pd
import numpy as np
import doctest

from texthero.lang.hero_zh import preprocessing, stopwords
from ... import PandasTestCase


"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(preprocessing))
    return tests


class TestPreprocessing(PandasTestCase):
    """
    Remove whitespace.
    """

    def test_remove_whitespace(self):
        s = pd.Series("早上好啊，\n\t我的朋友。今天我要去吃     KFC。")
        s_true = pd.Series("早上好啊， 我的朋友。今天我要去吃 KFC。")
        self.assertEqual(preprocessing.remove_whitespace(s), s_true)

    """
    Test pipeline.
    """

    def test_pipeline_stopwords(self):
        s = pd.Series("语言是人类区别其他动物的本质特性。\t@中国NLP第一大师\n#如何定义NLP 为什么呢？")
        s_true = pd.Series("语言是人类区别其他动物的本质特性。     为什么呢？")
        pipeline = [
            preprocessing.remove_whitespace,
            preprocessing.remove_hashtags,
            preprocessing.remove_tags,
        ]
        self.assertEqual(preprocessing.clean(s, pipeline=pipeline), s_true)

    """
    Test remove html tags
    """

    def test_remove_html_tags(self):
        s = pd.Series("<html> 中国新闻网 <br>体育</br> 标记<html> &nbsp;")
        s_true = pd.Series(" 中国新闻网 体育 标记 ")
        self.assertEqual(preprocessing.remove_html_tags(s), s_true)

    """
    Text tokenization
    """

    def test_tokenize(self):
        s = pd.Series("我昨天吃烤鸭去了。")
        s_true = pd.Series([["我", "昨天", "吃", "烤鸭", "去", "了", "。"]])
        self.assertEqual(preprocessing.tokenize(s), s_true)

    def test_tokenize_multirows(self):
        s = pd.Series(["今天天气真好", "明天会怎样呢"])
        s_true = pd.Series([["今天天气", "真", "好"], ["明天", "会", "怎样", "呢"]])
        self.assertEqual(preprocessing.tokenize(s), s_true)

    """
    Has content
    """

    def test_has_content(self):
        s = pd.Series(["哈哈", np.nan, "\t\n", " ", "", "这有点东西", None])
        s_true = pd.Series([True, False, False, False, False, True, False])
        self.assertEqual(preprocessing.has_content(s), s_true)

    """
    Test remove urls
    """

    def test_remove_urls(self):
        s = pd.Series("http://tests.com http://www.tests.com")
        s_true = pd.Series("   ")
        self.assertEqual(preprocessing.remove_urls(s), s_true)

    def test_remove_urls_https(self):
        s = pd.Series("https://tests.com https://www.tests.com")
        s_true = pd.Series("   ")
        self.assertEqual(preprocessing.remove_urls(s), s_true)

    def test_remove_urls_multiline(self):
        s = pd.Series("https://tests.com \n https://tests.com")
        s_true = pd.Series("  \n  ")
        self.assertEqual(preprocessing.remove_urls(s), s_true)

    """
    Test replace and remove tags
    """

    def test_replace_tags(self):
        s = pd.Series("你好@马丁123abc佩奇，我要把你取关了。")
        s_true = pd.Series("你好TAG，我要把你取关了。")

        self.assertEqual(preprocessing.replace_tags(s, symbol="TAG"), s_true)

    def test_remove_tags(self):
        s = pd.Series("你好@马丁123abc佩奇，我要把你取关了。")
        s_true = pd.Series("你好 ，我要把你取关了。")

        self.assertEqual(preprocessing.remove_tags(s), s_true)

    """
    Test replace and remove hashtags
    """

    def test_replace_hashtags(self):
        s = pd.Series("语言是人类区别其他动物的本质特性。#NLP百科大全")
        s_true = pd.Series("语言是人类区别其他动物的本质特性。HASHTAG")

        self.assertEqual(preprocessing.replace_hashtags(s, symbol="HASHTAG"), s_true)

    def test_remove_hashtags(self):
        s = pd.Series("语言是人类区别其他动物的本质特性。#NLP百科大全")
        s_true = pd.Series("语言是人类区别其他动物的本质特性。 ")

        self.assertEqual(preprocessing.remove_hashtags(s), s_true)
