from typing import List

from pathlib import Path
from datetime import datetime
import logging
from logging import Formatter, handlers, getLogger
import inspect
import os

import coloredlogs


LOG_FILE_NAME = f'app_{datetime.strftime(datetime.now(), "%Y%m%d")}.log'  # logファイルの名前

FIELD_STYLES = {
    'asctime': {'color': 'green'},
    'hostname': {'color': 'magenta'},
    'levelname': {'color': 'blue', 'bold': True},
    'name': {'color': 'blue'},
    'programname': {'color': 'cyan'}}

LEVEL_STYLES = {
    'critical': {'color': 'red', 'bold': True},
    'error': {'color': 'red'},
    'warning': {'color': 'yellow'},
    'notice': {'color': 'magenta'},
    'info': {'color': 'green'},
    'debug': {'color': 'green'},
    'spam': {'color': 'green', 'faint': True},
    'success': {'color': 'green', 'bold': True},
    'verbose': {'color': 'blue'}}


class Logger:
    """
    Attributes
    ----------
    log_level : str
        追跡するログレベル, e.g. 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
    log_stdout : bool
        ログを標準出力するか否か
    save_log_dir : Path
        ログを書き出すディレクトリ

    Methods
    -------
    debug -> None
        debugレベルログを出力
    info -> None
        infoレベルログを出力
    warn -> None
        warnレベルログを出力
    error -> None
        errorレベルログを出力
    remove_oldlog -> None
        ログディレクトリにmax_log_num以上logファイルがある場合、最も古いlogファイルを消去
    """
    def __init__(
            self,
            *,
            log_level: str = 'DEBUG',
            log_stdout: bool = True,
            save_log_dir: str = 'logs'):
        self.log_dir: Path = Path(save_log_dir)
        self.log_backupcount: int = 3
        os.makedirs(self.log_dir, exist_ok=True)  # logフォルダが無ければ作成
        log_file_path: Path = self.log_dir/LOG_FILE_NAME
        coloredlogs.CAN_USE_BOLD_FONT = True
        coloredlogs.DEFAULT_FIELD_STYLES = FIELD_STYLES
        coloredlogs.DEFAULT_LEVEL_STYLES = LEVEL_STYLES

        # ロガー生成
        caller_func_name: str = inspect.stack()[1].filename.split('/')[-1]  # 呼び出し元関数名
        self.logger: logging.Logger = getLogger(caller_func_name)
        self.logger.setLevel(log_level)
        formatter = Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)7s %(message)s [%(name)s:%(lineno)d]",
            datefmt="%Y/%m/%d %H:%M:%S")
        # サイズローテーション
        handler = handlers.RotatingFileHandler(
            filename=log_file_path,
            encoding='UTF-8',
            maxBytes=2**24,  # 16MB
            backupCount=self.log_backupcount)
        # ログファイル設定
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # 標準出力用 設定
        if log_stdout:
            coloredlogs.install(level=log_level, logger=self.logger, fmt=formatter._fmt, datefmt=formatter.datefmt)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg, stacklevel=2)

    def info(self, msg: str) -> None:
        self.logger.info(msg, stacklevel=2)

    def warn(self, msg: str) -> None:
        self.logger.warning(msg, stacklevel=2)

    def error(self, msg: str, *, exc_info: bool = True) -> None:
        self.logger.error(msg, exc_info=exc_info, stacklevel=2)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg, stacklevel=2)

    def remove_oldlog(self, *, max_num_log: int = 100) -> None:
        """ログディレクトリにmax_log_num以上logファイルがある場合、最も古いlogファイルを消去

        Parameters
        ----------
        max_log_num : int, optional
            logファイルの最大件数, by default 100
        """
        logs: List[str] = self.log_dir.glob('*.log')
        if len(logs) > max_num_log:
            # {logname}_yyyymmdd.log -> yyyymmdd
            log_name_pairs = [[log, datetime.strptime(log[-12:-4], '%Y%m%d')] for log in logs]
            log_name_pairs = sorted(log_name_pairs, key=lambda s: s[1])
            remove_log_path = log_name_pairs[0][0]  # 最も古いlogファイル
            os.remove(remove_log_path)
            self.info(f'remove {remove_log_path}')
            # ローテーションされたlogファイル (e.g. {logname}_yyyymmdd.log.1) がある場合、それらも削除する
            for i in range(1, self.log_backupcount+1):
                remove_rotating_log_path = remove_log_path + f'.{i}'
                if os.path.exists(remove_rotating_log_path):
                    os.remove(remove_rotating_log_path)
                    self.info(f'removed {remove_rotating_log_path}')
