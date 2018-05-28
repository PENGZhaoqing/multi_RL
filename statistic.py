from io import open
import matplotlib.pyplot as plt
import numpy as np
import re


def range_x(data):
    return np.array(range(len(data))) * 0.5


# greeen, pink, blue,red,purple
color_maps = ['#009E73', '#c46ea0', '#0B77B5', '#ff0066', '#a64dff', '#ff9900']


def smooth_online(data):
    return data.reshape(80, -1, 32).mean(axis=1)


class Statistic(object):
    def __init__(self, file_name, name, agent_num, envs_num):
        self.reward = []
        self.rewards = []
        self.collision = []
        self.collisions = []
        self.values = []
        self.filename = file_name
        self.name = name
        self.agent_num = agent_num
        self.envs_num = envs_num

        self.mean_reward = None
        self.mean_steps = None
        self.mean_collision = None
        self.mean_loss = None

        with open(self.filename, 'r') as text_file:
            lines = text_file.readlines()
            for line in lines:
                columns = line.split(',')
                if len(columns) == 43:
                    self.reward.append(columns[3].split(':')[1])
                    self.collision.append(columns[4].split(':')[1])
        self.reward = np.array(self.reward, dtype=float)
        self.collision = np.array(self.collision, dtype=float)

        with open(self.filename, 'r') as text_file:
            lines = text_file.read().replace('\n', '')
            reward = re.findall('\[.*?\]', lines)
            for line in reward:
                reward_list = line.replace('[', '').replace(']', '').split()
                self.rewards.append(reward_list)
            tmp = re.findall('\(.*?\)', lines)
            for line in tmp:
                tmp_list = line.replace('(', '').replace(')', '').replace(',', '').split()
                if len(tmp_list) == self.envs_num:
                    self.collisions.append(tmp_list)
                elif len(tmp_list) == self.agent_num:
                    self.values.append(tmp_list)

        self.rewards = np.array(self.rewards, dtype=float)
        self.collisions = np.array(self.collisions, dtype=float)
        self.values = np.array(self.values, dtype=float)


def confidence_band(stats, data_type='reward'):
    x_axis = []
    means = []
    stds = []

    for idx in range(len(stats)):
        mean, std = None, None
        if data_type == 'reward':
            x_axis.append(range_x(stats[idx].rewards))
            mean, std = mean_std(stats[idx].rewards)
        elif data_type == 'collision':
            x_axis.append(range_x(stats[idx].collisions))
            mean, std = mean_std(stats[idx].collisions)
        elif data_type == 'value':
            x_axis.append(range_x(stats[idx].values))
            mean, std = mean_std(stats[idx].values)
        means.append(mean)
        stds.append(std)

    for idx in range(len(stats)):
        plt.plot(x_axis[idx], means[idx], 'k', color=color_maps[idx], label=stats[idx].name)
        plt.fill_between(x_axis[idx], means[idx] - stds[idx], means[idx] + stds[idx],
                         alpha=0.1, edgecolor=color_maps[idx], facecolor=color_maps[idx], linewidth=1)

    plt.legend(loc=2, labelspacing=0, fancybox=True, framealpha=0.5)
    if data_type == 'collision':
        axes = plt.gca()
        # axes.set_ylim([0, 40])
        # axes.set_ylim([-20, 60])
    elif data_type == 'reward':
        axes = plt.gca()
        # axes.set_ylim([-10, 60])
    elif data_type == 'value':
        axes = plt.gca()
        # axes.set_ylim([-50, 50])
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def mean_std(data):
    return np.mean(data, axis=1), np.std(data, axis=1)


commNet = Statistic("log/hunterworld/CommNet_agents_5_dynamic_True_04:27:19:38.log", 'CommNet', agent_num=5,
                    envs_num=32)
IL = Statistic("log/hunterworld/IL_agents_5_dynamic_True_04:27:20:18.log", 'IL', agent_num=5, envs_num=32)
VAIN = Statistic("log/hunterworld/VAIN_agents_5_dynamic_True_04:27:20:56.log", 'VAIN', agent_num=5, envs_num=32)
DIAL = Statistic("log/hunterworld/DIAL_agents_5_dynamic_True_04:27:22:30.log", 'DIAL', agent_num=5, envs_num=32)
# LSTM = Statistic("log/hunterworld-toxin/LSTM_agents_5_dynamic_True_04:26:02:42.log", 'LSTM', agent_num=5,
#                    envs_num=32)
ARMI = Statistic("log/hunterworld/ARMI_agents_5_dynamic_True_04:27:18:33.log", 'ARMI', agent_num=5,
                 envs_num=32)

stats = [commNet, IL, VAIN, DIAL, ARMI]

# reg = np.ones((80, 32))
# reg = reg * 5
# reg[:8] = 2
# reg[8:16] = 3
# reg[16:24] = 4
# for stat in stats:
#     stat.rewards = stat.rewards / reg

for stat in stats:
    print "%s,  %0.1f %0.1f,     %0.1f %0.1f,     %0.1f" % (
        stat.name, stat.rewards[-40:].mean(), stat.rewards[-40:].std(),
        stat.collisions[-40:].mean(), stat.collisions[-40:].std(),
        stat.rewards[-40:].mean() / stat.collisions[-40:].mean())

fig = plt.figure(1, figsize=(20, 8))

# plt.subplot(121)
confidence_band(stats, data_type='reward')
ax = plt.gca()
# ax.set_title("Rewards")
ax.set_facecolor('#EAEAF2')
plt.ylabel('Scores per agent')
plt.xlabel('Running epochs')
plt.grid(color='w', linestyle='-', linewidth=1)

# plt.subplot(122)
# confidence_band(stats, data_type='collision')
# ax = plt.gca()
# # ax.set_title("collisions")
# ax.set_facecolor('#EAEAF2')
# plt.ylabel('Collision')
# plt.xlabel('Running epochs')
# plt.grid(color='w', linestyle='-', linewidth=1)
#
# plt.subplot(133)
# confidence_band(stats, data_type='value')
# ax = plt.gca()
# # ax.set_title("values")
# ax.set_facecolor('#EAEAF2')
# plt.ylabel('Values')
# plt.xlabel('Running epochs')
# plt.grid(color='w', linestyle='-', linewidth=1)

# plt.subplot(224)
# confidence_band_online(smooth_online(toxin_IL_online.rewards), smooth_online(toxin_GL_online.rewards),
#                        smooth_online(toxin_PS_online.rewards), smooth_online(toxin_DIAL_online.rewards),
#                        smooth_online(toxin_ML_online.rewards),
#                        )
# ax = plt.gca()
# ax.set_title("Online updating methods in 2-10-5")
# ax.set_facecolor('#EAEAF2')
# plt.ylabel('Scores')
# plt.xlabel('Running epochs')
# plt.grid(color='w', linestyle='-', linewidth=1)

# fig.tight_layout()
# plt.setp(lines, color='b', linewidth=1.0)
# plt.title(r'$\sigma_i=15$')
plt.tight_layout(pad=10, w_pad=0, h_pad=0)
plt.show()
# plt.savefig('destination_path.svg', format='svg', dpi=1000)
