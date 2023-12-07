import time
import re
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class Email():
    def __init__(self, host="smtp.163.com", sender="xxx@163.com", passwd="xxx") -> None:
        self.host = host
        self.sender = sender
        self.passwd = passwd
    
    def send_email(self, receivers=["xxx@xx.com", "xx@xx.com"], subject="", body=""):
        """定义发送邮件的函数"""
        message = MIMEText(body, 'plain', 'utf-8')
        message['From'] = Header(self.sender, 'utf-8')
        message['To'] = ";".join(receivers)
        message['Subject'] = Header(subject, 'utf-8')
        try:
            smtpObj = smtplib.SMTP_SSL(self.host)
            smtpObj.login(self.sender, self.passwd)  # 发件人的邮箱和授权码
            smtpObj.sendmail(self.sender, receivers, message.as_string())
            logging.info("发送状态邮件成功！")
        except smtplib.SMTPException as e:
            logging.info(f"发送状态邮件失败！{e}")


class NvidiaSmiMonitor:
    def __init__(self, server_name, MEMORY_USAGE=10, interval_minutes=10):
        """
        args:
        server_name: 服务器名
        MEMORY_USAGE: G, 显存使用量比它小时，则认为空闲
        interval_minutes: 查询间隔时间
        """
        self.server_name = server_name
        self.MEMORY_USAGE = MEMORY_USAGE * 1000
        self.interval_minutes = interval_minutes
        self.last_free_count = -1
        self.email = Email()
    
    def run(self):
        while True:
            # 运行nvidia-smi命令，并捕获其输出
            process = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            output = process.stdout.decode('utf-8')
            # 使用正则表达式查找显存使用量
            matches = re.findall(r'(\d+)MiB /', output)
            if matches:
                logging.info(matches)
                # 计算空闲显卡数量
                free_count = sum([1 for m in matches if int(m) < self.MEMORY_USAGE])
                # 如果空闲显卡数量发生改变，则提醒用户
                if free_count != self.last_free_count:
                    # 发送邮件提醒
                    subject = self.server_name + f"空卡数量变更{self.last_free_count}-->{free_count}"
                    body = self.server_name + f"\n空闲显卡数量: {free_count}\n各卡已用显存分别是: {matches}"
                    self.email.send_email(subject=subject, body=body)
                    logging.info(f"{subject}, {body}")
                    self.last_free_count = free_count
                # 等待指定的时间间隔，再次运行监控程序
                time.sleep(self.interval_minutes * 60)


if __name__ == "__main__":
    monitor = NvidiaSmiMonitor(server_name="hostxx", MEMORY_USAGE=18)
    monitor.run()
