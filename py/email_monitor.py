import smtplib
from email.mime.text import MIMEText
import subprocess
import time
import re
from email.header import Header


class Email():
    def __init__(self) -> None:
        self.host = "smtp.163.com"
        self.sender = "xx@163.com"
        self.passwd = "password"
        self.receivers = ["xx@126.com"]
    
    # 定义发送邮件的函数
    def send_email(self, host=None, sender=None, receivers=None, subject="", body=""):
        if not host:
            host = self.host
        if not sender:
            sender = self.sender
        if not receivers:
            receivers = self.receivers
        message = MIMEText(body, 'plain', 'utf-8')
        message['From'] = Header(sender, 'utf-8')
        message['To'] = ";".join(receivers)
        message['Subject'] = Header(subject, 'utf-8')
        try:
            smtpObj = smtplib.SMTP_SSL(self.host)
            smtpObj.login(sender, self.passwd)  # 发件人的邮箱和授权码
            smtpObj.sendmail(sender, receivers, message.as_string())
            print("发送状态邮件成功！", flush=True)
        except smtplib.SMTPException as e:
            print("发送状态邮件失败！", e, flush=True)


class NvidiaSmiMonitor:
    def __init__(self, MEMORY_USAGE=10000, interval_minutes=10):
        self.MEMORY_USAGE = MEMORY_USAGE  # MiB, 显存使用量比它小时，则认为空闲
        self.last_free_count = -1
        self.interval_minutes = interval_minutes
    
    def run(self):
        email = Email()
        while True:
            # 运行nvidia-smi命令，并捕获其输出
            process = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            output = process.stdout.decode('utf-8')
            # 使用正则表达式查找显存使用量
            matches = re.findall(r'(\d+)MiB /', output)
            # print(matches, flush=True)
            # 计算空闲显卡数量
            free_count = sum([1 for m in matches if int(m) < self.MEMORY_USAGE])
            # 如果空闲显卡数量发生改变，则提醒用户
            if free_count != self.last_free_count:
                # 发送邮件提醒
                subject = "空闲显卡数量变更提醒"
                body = f"host34\n空闲显卡数量: {free_count}\n各卡已用显存分别是: {matches}"
                email.send_email(subject=subject, body=body)
                print(subject, body, flush=True)
                self.last_free_count = free_count
            # 等待指定的时间间隔，再次运行监控程序
            time.sleep(self.interval_minutes * 60)


if __name__ == "__main__":
    monitor = NvidiaSmiMonitor()
    monitor.run()
