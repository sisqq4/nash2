from __future__ import annotations

import copy
import math as m
import random as rd
from typing import List, Optional, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

# 这些依赖在你的工程里应当已有实现/定义：
# - tool：提供 change_axis/torad/todeg/norm/distance/angle3/angle4 等
# - RPM：经验回放（或你工程里的 RPM 类）
# - Missile：导弹类（仅用于类型注解）
# - action_space：提供 get_simple/get/get_complex（或同名接口）
# - plane_action_dict_simple：简单动作字典（用于 get_simple）
import tool  # type: ignore
import action_space  # type: ignore

try:
    from rpm import RPM  # type: ignore
except Exception:  # 兼容：如果 RPM 在其他模块，按你的工程改
    RPM = Any  # type: ignore

try:
    from missile import Missile  # type: ignore
except Exception:
    Missile = Any  # type: ignore

try:
    from action_space import plane_action_dict_simple  # type: ignore
except Exception:
    plane_action_dict_simple = {}  # type: ignore


class Agent:
    def __init__(self, no, position, angle, speed, belong, net, xyz_axis=True) -> None:
        self.no = no
        if xyz_axis:
            self.position = copy.deepcopy(position)  # x,y,z
            self.angle = copy.deepcopy(angle)        # roll,pitch,yaw
            self.speed = speed
        else:
            temp, self.speed = tool.change_axis(position + angle, None, x2north=False)
            self.position = temp[:3]
            self.angle = temp[3:]
            # print('self.angle', self.angle)
            # print('self.position', self.position)

        self.belong = belong
        self.net = net
        self.rpm = RPM()
        self.state = None
        self.next_state = None
        self.act_index = None
        self.act = None
        self.reward = 0

        # act_list 在 Plane 里会用到；这里不强制创建，避免改变你原始设计
        # 但封装接口会通过 getattr 兼容（见 pop_act/has_act/push_acts）

    # ---------------------------
    # act_list 队列封装接口（新增）
    # ---------------------------
    def has_act(self) -> bool:
        return bool(getattr(self, "act_list", []))

    def pop_act(self) -> Optional[list]:
        act_list = getattr(self, "act_list", None)
        if act_list is None or len(act_list) == 0:
            return None
        act = act_list[0]
        setattr(self, "act_list", act_list[1:])
        return act

    def push_acts(self, acts: list, repeat: int = 1) -> None:
        # acts: 形如 [[...],[...]] 的动作序列；repeat>1 表示重复拼接
        if not hasattr(self, "act_list") or getattr(self, "act_list") is None:
            setattr(self, "act_list", [])
        cur = getattr(self, "act_list")
        for _ in range(repeat):
            cur.extend(copy.deepcopy(acts))
        setattr(self, "act_list", cur)

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    @x.setter
    def x(self, value):
        self.position[0] = value

    @y.setter
    def y(self, value):
        self.position[1] = value

    @z.setter
    def z(self, value):
        self.position[2] = value

    @property
    def roll(self):  # 存储角度，输出弧度
        return tool.torad(self.angle[0])

    @property
    def pitch(self):
        return tool.torad(self.angle[1])

    @property
    def yaw(self):
        return tool.torad(self.angle[2])

    @roll.setter
    def roll(self, value):  # 输入弧度，存储角度
        self.angle[0] = tool.todeg(value)

    @pitch.setter
    def pitch(self, value):
        self.angle[1] = tool.todeg(value)

    @yaw.setter
    def yaw(self, value):
        self.angle[2] = tool.todeg(value)

    @property
    def north(self):
        return self.position[1]

    @property
    def sky(self):
        return self.position[2]

    @property
    def east(self):
        return self.position[0]

    @north.setter
    def north(self, value):
        self.position[1] = value

    @sky.setter
    def sky(self, value):
        self.position[2] = value

    @east.setter
    def east(self, value):
        self.position[0] = value

    @property
    def vnorth(self):
        temp = tool.change_axis(self.position + self.angle, self.speed, x2north=True)
        return temp[3]

    @property
    def vsky(self):
        temp = tool.change_axis(self.position + self.angle, self.speed, x2north=True)
        return temp[4]

    @property
    def veast(self):
        temp = tool.change_axis(self.position + self.angle, self.speed, x2north=True)
        return temp[5]

    @vnorth.setter
    def vnorth(self, value):
        temp = tool.change_axis(self.position + self.angle, self.speed, x2north=True)
        temp[3] = value
        temp, self.speed = tool.change_axis(temp, None, x2north=False)
        self.angle = temp[3:]

    @vsky.setter
    def vsky(self, value):
        temp = tool.change_axis(self.position + self.angle, self.speed, x2north=True)
        temp[4] = value
        temp, self.speed = tool.change_axis(temp, None, x2north=False)
        self.angle = temp[3:]

    @veast.setter
    def veast(self, value):
        temp = tool.change_axis(self.position + self.angle, self.speed, x2north=True)
        temp[5] = value
        temp, self.speed = tool.change_axis(temp, None, x2north=False)
        self.angle = temp[3:]

    def sample(self):
        self.act_index = self.net.sample(self.state)
        act = self.action_space()[self.act_index].tolist()
        return act

    def predict(self):
        self.act_index = self.net.predict(self.state)
        act = self.action_space()[self.act_index].tolist()
        return act

    def __repr__(self) -> str:
        return str(self.__dict__)

    def pnorm(self, boun):
        res = []
        for i in range(3):
            res.append(tool.norm(self.position[i], boun[i][0], boun[i][1]))
        return np.array(res)

    def anorm(self):
        res = []
        for i in range(3):
            res.append(tool.norm(self.angle[i], -180, 180))
        return np.array(res)

    def apnorm(self, boun):
        return np.append(self.pnorm(boun), self.anorm())

    def vapnorm(self, boun):
        return np.append(self.apnorm(boun), np.array([tool.norm(self.speed, 0, 2000)]))

    def action_space(self) -> np.ndarray:
        raise ValueError("err->virtual fun!")

    def step(self):
        raise ValueError("err->virtual fun!")

    def distance(self, agent):
        return tool.distance(self.position, agent.position)

    def hit_boundary(self, boundary):
        for i in range(3):
            if self.position[i] <= boundary[i][0] or self.position[i] >= boundary[i][1]:
                return True
        return False

    def get_reward(self, boundary):
        reward = 0
        r = boundary[0][1] - boundary[0][0]
        for i in range(3):
            min_gap = 999999999
            for boun in boundary[i]:
                gap = abs(self.position[i] - boun)
                if gap <= min_gap:
                    min_gap = gap
            reward = (min_gap - r / 2) / (r / 2)
        return reward

    def get_Rho(self):
        rho0 = 1.225
        T0 = 288.15
        if self.z <= 11000:
            T = T0 - 0.0065 * self.z
            Rho = rho0 * (T / T0) ** 4.25588
        elif self.z > 11000 and self.z <= 20000:
            T = 216.65
            Rho = 0.36392 * m.exp((-self.z + 11000) / 6341.62)
        else:
            T = 216.65 + 0.001 * (self.z - 20000)
            Rho = 0.088035 * (T / 216.65) ** -35.1632
        return Rho

    def get_friction(self):
        Rho = self.get_Rho()
        speed = self.speed
        friction = 0.5 * Rho * speed ** 2 * 2 * 0.015  # 阻力系数取0.28 迎风面积150cm^2
        return friction / 157  # 使用AIM120的质量

    def save(self, filename):
        self.net.save(filename)

    def load(self, fname):
        self.net.load(fname)


class Plane(Agent):
    def __init__(self, no, position, angle, speed, belong, missile_list, net, xyz_axis=True) -> None:
        super(Plane, self).__init__(no, position, angle, speed, belong, net, xyz_axis)
        self.missile_list: List[Missile] = missile_list
        self.action_step_count = 20
        self.act_list = []
        self.max_speed = 600
        self.ismax = False
        self.dis_ref = -1
        self.min_speed = 100
        self.act_list = []
        self.s_flag = 0
        self.start_action = 0  # 增加
        self.action_flag = 0

    def step(self, nx, nz, roll, g, dt):
        self.roll = roll
        dv = g * (nx - m.sin(self.pitch))  # 速度标量加速度
        dpitch = (g / self.speed) * (nz * m.cos(self.roll) - m.cos(self.pitch))  # 俯仰角速度
        dyaw = g * nz * m.sin(self.roll) / (self.speed * m.cos(self.pitch))  # 偏航角速度

        self.speed += dv * dt
        if self.speed >= self.max_speed:
            self.speed = self.max_speed
            self.ismax = True
        else:
            self.ismax = False

        if self.speed <= self.min_speed:
            self.speed = self.min_speed

        self.pitch += dpitch * dt
        self.yaw += dyaw * dt

        # 90度航迹角突变
        if self.pitch > 0.5 * m.pi:
            self.pitch = m.pi - self.pitch
            if self.yaw > 0:
                self.yaw = self.yaw - m.pi
            else:
                self.yaw = self.yaw + m.pi

            if self.roll > 0:
                self.roll = self.roll - m.pi
            else:
                self.roll = self.roll + m.pi

        elif self.pitch < -0.5 * m.pi:
            self.pitch = -m.pi - self.pitch
            if self.yaw > 0:
                self.yaw = self.yaw - m.pi
            else:
                self.yaw = self.yaw + m.pi

            if self.roll > 0:
                self.roll = self.roll - m.pi
            else:
                self.roll = self.roll + m.pi

        self.x += self.speed * m.cos(self.pitch) * m.cos(self.yaw) * dt
        self.y += self.speed * m.cos(self.pitch) * m.sin(self.yaw) * dt
        self.z += self.speed * m.sin(self.pitch) * dt  # 高度轴

    def action_space_simple(self):
        return action_space.get_simple()

    def action_space(self):
        return action_space.get()

    def action_space_complex(self):
        return action_space.get_complex()

    def sample(self, complex=-1):
        if complex == 1:
            if len(self.act_list) == 0:
                self.act_index = self.net.sample(self.state)
                self.act_list = self.action_space_complex()[self.act_index].tolist()
            act = self.act_list[0]
            self.act_list = self.act_list[1:]
            return act, self.act_index

        elif complex == -1:
            self.act_index = self.net.sample(self.state)
            act = self.action_space_simple()[self.act_index]
            return act, self.act_index

        else:
            self.act_index = self.net.sample(self.state)
            act = self.action_space()[self.act_index].tolist()
            return act, self.act_index

    def overload(self):  # 过载按照海拔高度限制
        if self.sky >= 25000:
            return 2
        elif self.sky >= 15000 and self.sky < 25000:
            return 4 - 2 * (self.sky - 15000) / 10000
        elif self.sky >= 12000 and self.sky < 15000:
            return 5 - (self.sky - 12000) / 3000
        elif self.sky >= 9000 and self.sky < 12000:
            return 6 - (self.sky - 9000) / 3000
        elif self.sky >= 7000 and self.sky < 9000:
            return 7 - (self.sky - 7000) / 2000
        elif self.sky >= 6000 and self.sky < 7000:
            return 7.5 - (self.sky - 6000) / 2000
        elif self.sky >= 3000 and self.sky < 6000:
            return 9 - (self.sky - 3000) / 2000
        else:
            return 9

    def predict(self, complex=-1):
        if complex == 1:
            if len(self.act_list) == 0:
                self.act_index = self.net.predict(self.state)
                self.act_list = self.action_space_complex()[self.act_index].tolist()
            act = self.act_list[0]
            self.act_list = self.act_list[1:]
            return act, self.act_index

        elif complex == -1:
            self.act_index = self.net.predict(self.state)
            act = self.action_space_simple()[self.act_index].tolist()
            return act, self.act_index

        else:
            self.act_index = self.net.predict(self.state)
            act = self.action_space()[self.act_index].tolist()
            return act, self.act_index

    def predict_state(self, missile, plane=None):
        if missile == None:
            dx = plane.north - self.north
            dz = plane.sky - self.sky
            dy = plane.east - self.east
            theta = tool.angle3([dx, dy], [self.vnorth, self.veast])
            result = tool.angle4([dx, 0, dy], [self.vnorth, 0, self.veast])  # result[1]为正表示弹目连线在飞机的左边，向左转；负相反
            sign = 1

            act = self.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            # 紧急避险
            if self.sky < 2000 and self.angle[1] < 0:
                return [0, 4, 0, 0]

            if self.angle[1] > 0 or self.sky - plane.sky > 2000:
                sign = -1
            elif self.angle[1] < 0 or self.sky - plane.sky < -3000:
                sign = 1
            else:
                sign = 0

            if theta < 10:
                if self.sky - plane.sky > 2000:
                    if self.angle[1] > -30:
                        return [0, 0, 0, 0]
                    else:
                        return [0.5, 1, 0, 0]

                elif self.sky - plane.sky < -3000:
                    if self.angle[1] < 30:
                        return [0, 3, 0, 0]
                    else:
                        return [0.5, 1, 0, 0]

                else:
                    if self.angle[1] > 5 or self.angle[1] < -5:  # 向下俯冲回正
                        return [0, sign * 5, 0, -1]
                    else:
                        return [1, 1, 0, 0]

            elif result[1] > 0:  # result[1]为正表示弹目连线在飞机的左边，向左转
                if sign == 0:
                    return [0, 5, -m.acos(1 / 5), 0]
                else:
                    self.act_list = [[0, sign * 5, -sign * m.acos(1 / 2), -1]] * 50

            else:
                if sign == 0:
                    return [0, 5, m.acos(1 / 5), 0]
                else:
                    self.act_list = [[0, sign * 5, sign * m.acos(1 / 2), -1]] * 50

            act = self.pop_act()
            return act if act is not None else [0, 1, 0, 0]

        else:
            dx = self.north - missile[0]
            dz = self.sky - missile[1]
            dy = self.east - missile[2]
            theta = tool.angle3([self.vnorth, 0, self.veast], [dx, 0, dy])  # theta是飞机速度向量与导弹速度夹角
            result = tool.angle4([self.vnorth, 0, self.veast], [dx, 0, dy])  # result是飞机速度向量与导弹指向目标的距离向量的叉乘
            sign = 1

            act = self.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            # 退化为水平至尾
            if self.angle[1] < 0:
                return [0, 5, 0, 0]
            elif self.angle[1] > 10:
                return [0, -3, 0, 0]

            if theta < 10:
                if self.speed > 280:
                    return [-1, 1, 0, 0]
                else:
                    return [1, 1, 0, 0]

            elif result[1] > 0:
                if theta >= 160 and theta <= 180:
                    return [0, 6, m.acos(1 / 6), -1]
                else:
                    return [0, 6, m.acos(1 / 6), -1]

            else:
                if theta >= 160 and theta <= 180:
                    return [0, 6, -m.acos(1 / 6), -1]
                else:
                    return [0, 6, -m.acos(1 / 6), -1]

    def predict_state_simple(self, missile, plane=None):
        distance = 0

        if missile == None:
            distance = tool.distance(plane.position, self.position)
            dx = plane.position[0] - self.position[0]
            dy = plane.position[1] - self.position[1]
            dz = plane.position[2] - self.position[2]

            if dx > 0:
                pyaw = m.degrees(m.atan(dy / dx))
            elif dx == 0:
                pyaw = 90 if dy > 0 else -90
            else:
                pyaw = m.degrees(m.atan(dy / dx))
                pyaw = pyaw + 180 if pyaw <= 0 else pyaw - 180

            dyaw = (pyaw - self.angle[2] + 180) % 360 - 180
            dline = np.linalg.norm([dx, dy])

            if dline == 0:
                ppitch = 90 if dz >= 0 else -90
            else:
                ppitch = m.degrees(m.atan(dz / dline))

            dpitch = ppitch - (-self.angle[1])

            if dpitch > 5:
                ny = -4
            elif dpitch < -5:
                ny = 4
            else:
                ny = 1

            if -5 < dyaw < 5:
                return [0, 1, 0, -1]

        dx = missile[0] - self.position[0]
        dy = missile[1] - self.position[1]
        distance = np.linalg.norm([dx, dy])

        if dx > 0:
            pyaw = m.degrees(m.atan(dy / dx))
        elif dx == 0:
            pyaw = 90 if dy > 0 else -90
        else:
            pyaw = m.degrees(m.atan(dy / dx))
            pyaw = pyaw + 180 if pyaw <= 0 else pyaw - 180

        # print(dx, dy, pyaw)
        dyaw = ((180 + pyaw) - self.angle[2] + 180) % 360 - 180

        if self.z > 3000:
            ny = 9 if self.angle[1] < 45 else 1
        else:
            ny = -9 if self.angle[1] > 0 else 1

        act = self.pop_act()
        if act is not None:  # 把连续的机动动作做完
            return act

        elif distance > 0:
            if -5 < dyaw < 5:
                return [0, 1, 0, -1]
            elif dyaw >= 5:
                return [0, 6, m.acos(1 / 6), -1]  # 左俯冲
            else:
                return [0, 6, -m.acos(1 / 6), -1]  # 右俯冲

        else:
            t = rd.randint(0, 4)
            if (t == 0):
                self.act_list = [[3, -20, m.acos(1 / 10), 0]] * 30 + [[1, -30, m.acos(1 / 10), 0]] * 30
            elif (t == 1):
                self.act_list = [[0, -20, 0, -1]] * 20  # 下俯冲
            elif (t == 2):
                self.act_list = [[0, 20, m.acos(1 / 20), 0]] * 20
            elif (t == 3):
                self.act_list = [[0, 20, -m.acos(1 / 20), 0]] * 20
            else:
                self.act_list = [[0, 1, m.pi * (i / 50), 0] for i in range(60)] + [[0, 10, m.pi, -1]] * 30

            self.act_index = 29
            act = self.pop_act()
            return act if act is not None else [0, 1, 0, 0]

        # target.ref_yaw = target.yaw
        # target.s_flag = 0
        return [0, 1, 0, 0]


def professional_action(index, missile, target: Plane, plane=None):
    if plane == None:
        # 此时变高至尾
        if index == 0 or index == 2:
            dx = target.north - missile[0]
            dz = target.sky - missile[1]
            dy = target.east - missile[2]
            theta = tool.angle3([target.vnorth, 0, target.veast], [dx, 0, dy])  # theta是飞机速度向量与导弹速度夹角
            result = tool.angle4([target.vnorth, 0, target.veast], [dx, 0, dy])  # result[1]为正表示弹目连线在飞机的左边
            sign = 1

            act = target.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            if target.sky - missile[1] <= 100:
                if target.sky > 5000:
                    sign = -1 if target.angle[1] > -60 else 1
                else:
                    sign = 1 if target.angle[1] < 0 else -1

            if theta < 10:
                if target.angle[1] > 5 or target.angle[1] < -5:  # 向下俯冲回正
                    return [0, sign * 6, 0, -1]
                else:
                    return [5, 1, 0, 0]

            # （承接：导弹比飞机高时的分支末尾，继续补全 result[1] 的左右转逻辑）
            elif result[1] > 0:  # result[1]为正表示弹目连线在飞机的左边，向左转
                if theta <= 180 and theta > 160:  # 此时迎头，减速置尾
                    target.act_list = [[-0.5, sign * 6, sign * m.acos(1 / 2), -1]] * 200
                else:
                    target.act_list = [[1, sign * 6, sign * m.acos(1 / 2), -1]] * 200
            else:
                if theta <= 180 and theta > 160:  # 此时迎头，减速置尾
                    target.act_list = [[-0.5, sign * 6, -sign * m.acos(1 / 2), -1]] * 200
                else:
                    target.act_list = [[1, sign * 6, -sign * m.acos(1 / 2), -1]] * 200

            act = target.pop_act()
            return act if act is not None else [5, 1, 0, 0]

        # -------------------------
        # 此时水平置尾
        # -------------------------
        elif index == 1:
            dx = target.north - missile[0]
            dz = target.sky - missile[1]
            dy = target.east - missile[2]
            theta = tool.angle3([target.vnorth, 0, target.veast], [dx, 0, dy])
            result = tool.angle4([target.vnorth, 0, target.veast], [dx, 0, dy])  # result[1]为正表示弹目连线在飞机的左边，向左转，负相反

            act = target.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            if theta < 10:
                target.act_list = [[4, 1, 0, -1]] * 200
            elif result[1] > 0:
                if theta >= 160 and theta <= 180:
                    target.act_list = [[-0.5, 6, m.acos(1 / 6), -1]] * 200  # 克服重力1g过载需要7个向上过载
                else:
                    target.act_list = [[1, 6, m.acos(1 / 6), -1]] * 200  # 克服重力1g过载需要7个向上过载
            else:
                if theta >= 160 and theta <= 180:
                    target.act_list = [[-0.5, 6, -m.acos(1 / 6), -1]] * 200  # 克服重力1g过载需要7个向上过载
                else:
                    target.act_list = [[1, 6, -m.acos(1 / 6), -1]] * 200  # 克服重力1g过载需要7个向上过载

            act = target.pop_act()
            return act if act is not None else [4, 1, 0, -1]

        # -------------------------
        # 下破S与殷麦曼
        # -------------------------
        elif index == 20:
            act = target.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            # 先做姿态回正（偏离大时直接返回修正动作；回正后继续进入机动状态机）
            if target.angle[1] > 5:  # 向上爬升回正
                return [0, -5, 0, -1]
            elif target.angle[1] < -5:  # 向下俯冲回正
                return [0, 7, 0, -1]
            else:
                pass

            if target.start_action == 0:  # 是否正在做机动
                if target.sky - missile[1] < 0:
                    target.action_flag = 1  # S
                else:
                    target.action_flag = 0  # 因麦曼

            if target.action_flag:  # 破S
                target.start_action = 1
                if target.s_flag == 0:
                    target.act_list = [[0, 1, m.pi * (i / 100), 0] for i in range(100)] + [[0, 9, m.pi, 0]] * 100
                    target.ref_yaw = target.yaw  # 用于记录飞机航向是否突变
                    target.s_flag = 1
                else:
                    if target.angle[1] < 0:
                        if target.ref_yaw * target.yaw >= 0:
                            # print('突变前')
                            return [0, 9, m.pi, 0]
                        else:
                            # print('突变后')
                            return [0, 9, 0, 0]
                    else:
                        target.act_list = [[0, 1, 0, 0]] * 100
                        target.ref_yaw = target.yaw
                        # print('结束s_flag',target.s_flag)
                        target.s_flag = 0
                        target.start_action = 0  # 破S结束

                act = target.pop_act()
                return act if act is not None else [0, 1, 0, 0]

            else:
                target.start_action = 1
                if target.s_flag == 0:
                    target.act_list = [[0, 9, 0, 0]] * 100
                    target.ref_yaw = target.yaw  # 用于记录飞机航向是否突变
                    target.s_flag = 1
                    # print('start,s_flag',target.s_flag,target.ref_yaw)
                else:
                    if target.angle[1] > 0:
                        if target.ref_yaw * target.yaw > 0:
                            # print('突变前')
                            return [0, 9, 0, 0]
                        else:
                            # print('突变后',target.yaw,target.ref_yaw)
                            return [0, 9, m.pi, 0]
                    else:
                        target.act_list = [[0, 1, m.pi * ((100 - i) / 100), 0] for i in range(100)] + [[5, 1, 0, 0]] * 100
                        target.ref_yaw = target.yaw
                        # print('结束s_flag',target.s_flag)
                        target.s_flag = 0
                        target.start_action = 0  # 破S结束

                act = target.pop_act()
                return act if act is not None else [0, 1, 0, 0]

        # 自动平飞（兜底）
        else:
            if target.angle[1] > 5:  # 向上爬升回正
                return [0, -5, 0, -1]
            elif target.angle[1] < -5:  # 向下俯冲回正
                return [0, 7, 0, -1]
            else:
                return [4, 1, 0, 0]

    # =========================
    # 下面这一段是 profession_action() 的 plane != None 分支（追击/对机动）
    # =========================
    else:
        # ####################baseline####################
        # ################破S转水平####################
        # 水平追击
        if index == 0:
            dx = plane.north - target.north
            dz = plane.sky - target.sky
            dy = plane.east - target.east
            theta = tool.angle3([dx, dy], [target.vnorth, target.veast])
            result = tool.angle4([dx, 0, dy], [target.vnorth, 0, target.veast])  # result[1]为正表示弹目连线在载机的左边，向左转；负相反

            act = target.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            if theta < 10:
                target.act_list = [[3, 1, 0, -1]] * 100
            elif result[1] >= 0:  # result[1]为正表示目标机与载机连线在载机的右边，向右转
                target.act_list = [[2, 6, -m.acos(1 / 6), -1]] * 100  # 克服重力1g过载需要8个向上过载
            else:
                target.act_list = [[2, 6, m.acos(1 / 6), -1]] * 100  # 克服重力1g过载需要8个向上过载

            act = target.pop_act()
            return act if act is not None else [3, 1, 0, -1]

        elif index == 1:
            dx = plane.north - target.north
            dz = plane.sky - target.sky
            dy = plane.east - target.east
            theta = tool.angle3([dx, dy], [target.vnorth, target.veast])
            result = tool.angle4([dx, 0, dy], [target.vnorth, 0, target.veast])  # result[1]为正表示弹目连线在载机的左边，向左转；负相反
            sign = 1

            act = target.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            if dz < -2000:  # 让自身保持在比追击目标高2000左右高度
                sign = -1
            else:
                sign = 1

            if theta < 10:
                if dz < -2000 and dz > -5000:  # 高度差不多，平着飞就行
                    if target.angle[1] > 5:  # 回正
                        return [2, -3, 0, -1]
                    elif target.angle[1] < -5:
                        return [2, 3, 0, -1]
                    else:
                        target.act_list = [[3, 1, 0, -1]] * 100
                elif dz > -1500:  # 变高飞
                    if target.angle[1] - plane.angle[1] < 45:
                        return [2, 2, 0, -1]
                    else_toggle = False
                    if not else_toggle:
                        target.act_list = [[4, 1, 0, -1]] * 100
                else:
                    if target.angle[1] - plane.angle[1] > -10:
                        return [2, -1, 0, -1]
                    else:
                        target.act_list = [[2, 1, 0, -1]] * 100

            elif result[1] >= 0:
                target.act_list = [[-0.5, sign * 9, -sign * m.acos(1 / 7), -1]] * 100  # 9g过载转弯
            else:
                target.act_list = [[-0.5, sign * 9, sign * m.acos(1 / 7), -1]] * 100  # 9g过载转弯

            act = target.pop_act()
            return act if act is not None else [2, 1, 0, -1]

        # 变高追击(降高慢，比目标高1500)
        elif index == 2:
            dx = plane.north - target.north
            dz = plane.sky - target.sky
            dy = plane.east - target.east
            theta = tool.angle3([dx, dy], [target.vnorth, target.veast])
            result = tool.angle4([dx, 0, dy], [target.vnorth, 0, target.veast])  # result[1]为正表示弹目连线在载机的左边，向左转；负相反

            act = target.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            if dz < -1500:  # 让自身保持在比追击目标高1500左右高度
                sign = -1
            else:
                sign = 1

            if theta < 10:
                if dz <= -1500:  # 高度差不多，平着飞就行
                    if target.angle[1] > 0:  # 回正
                        return [2, -3, 0, -1]
                    elif target.angle[1] < 0:
                        return [2, 3, 0, -1]
                    else:
                        target.act_list = [[3, 1, 0, -1]] * 100
                else:
                    # 你原代码在此处截断；这里给一个保守兜底：保持轻微拉升/保持
                    target.act_list = [[2, 1, 0, -1]] * 100

            elif result[1] >= 0:
                target.act_list = [[-0.5, sign * 9, -sign * m.acos(1 / 7), -1]] * 100
            else:
                target.act_list = [[-0.5, sign * 9, sign * m.acos(1 / 7), -1]] * 100

            act = target.pop_act()
            return act if act is not None else [2, 1, 0, -1]

        # =========================
        # 经典回转机动（index == 10）
        # =========================
        elif index == 10:
            dx = plane.north - target.north
            dz = plane.sky - target.sky
            dy = plane.east - target.east

            theta = tool.angle3([dx, dy], [target.north, target.east])
            # result[1]：为正表示“目标-载机连线”在载机右侧（向右转），为负相反（向左转）
            result = tool.angle4([dx, 0, dy], [target.north, 0, target.east])

            act = target.pop_act()
            if act is not None:  # 把连续的机动动作做完
                return act

            # 若目标在上方（俯仰角为正），先做一个“向下修正”的动作（对应你图中的 return [1, -4, 0, 0]）
            if target.angle[1] > 0:
                return [1, -4, 0, 0]

            # 机头方位差较小：优先做俯仰修正/微调
            if theta < 10:
                if target.angle[1] > 5:     # 向上爬升回正
                    return [0, -5, 0, -1]
                elif target.angle[1] < 0:   # 向下俯冲回正
                    return [0, 5, 0, -1]
                else:
                    return [3, 1, 0, 0]

            # 方位差较大：进入持续“回转”动作序列
            if result[1] >= 0:
                target.act_list = [[-1, 5, -m.acos(1 / 5), -1]] * 100
            else:
                target.act_list = [[-1, 5,  m.acos(1 / 5), -1]] * 100

            act = target.pop_act()
            return act if act is not None else [3, 1, 0, 0]

        # =========
        # 自动平飞
        # =========
        else:
            if target.angle[1] > 5:       # 向上爬升回正
                return [0, -5, 0, -1]
            elif target.angle[1] < -5:    # 向下俯冲回正
                return [0, 7, 0, -1]
            else:
                return [4, 1, 0, 0]


# =========================
# 动作库导出（simple）
# =========================
def get_simple():
    res = []
    for key, value in plane_action_dict_simple.items():
        res.append(value)
    return np.array(res)
