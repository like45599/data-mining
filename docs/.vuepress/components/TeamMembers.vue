<template>
  <div class="team-container">
    <!-- <div class="team-title">About us</div> -->
    <div class="team-members">
      <div class="member-card" v-for="(member, index) in members" :key="index">
        <div v-if="member.avatar" class="member-avatar-img">
          <img :src="member.avatar" :alt="member.name" />
        </div>
        <div v-else class="member-avatar" :style="{ backgroundColor: member.color || '#4285F4' }">
          <span class="member-initial">{{ getInitial(member.name) }}</span>
        </div>
        <div class="member-name">{{ member.name }}</div>
        <div class="member-role" v-if="member.role">{{ member.role }}</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TeamMembers',
  data() {
    return {
      members: [
        { 
          name: '张鹏宇 Тимофей', 
          avatar: '/images/icons/zhang.jpg',
          color: '#4285F4'
        },
        { 
          name: '刘祺泉 Светослав', 
          avatar: '/images/icons/liu.jpg',
          color: '#EA4335'
        },
        { 
          name: '高力柯 Всеволод', 
          avatar: '/images/icons/gao.jpg',
          color: '#FBBC05'
        },
        { 
          name: '葛颂 Игнат', 
          avatar: '/images/icons/葛颂.jpg',
          color: '#34A853'
        },
        { 
          name: '王琦 Леонтий', 
          avatar: '/images/icons/王琦.jpg',
          color: '#8E44AD'
        }
      ]
    }
  },
  methods: {
    getInitial(name) {
      return name ? name.charAt(0) : '?';
    }
  }
}
</script>

<style scoped>
.team-container {
  margin: 30px 0;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.team-title {
  text-align: center;
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 20px;
  color: #2c3e50;
}

.team-members {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 20px;
}

.member-card {
  background-color: white;
  border-radius: 8px;
  padding: 20px;
  width: 160px;
  text-align: center;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: transform 0.3s, box-shadow 0.3s;
}

.member-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.member-avatar {
  margin: 0 auto 15px;
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.member-initial {
  color: white;
  font-size: 32px;
  font-weight: bold;
}

.member-avatar-img {
  margin: 0 auto 15px;
  width: 80px;
  height: 80px;
  border-radius: 50%;
  overflow: hidden;
}

.member-avatar-img img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.member-name {
  font-weight: bold;
  font-size: 16px;
  margin-bottom: 5px;
  color: #2c3e50;
}

.member-role {
  font-size: 14px;
  color: #7f8c8d;
}

@media (max-width: 768px) {
  .team-members {
    gap: 15px;
  }
  
  .member-card {
    width: calc(50% - 15px);
  }
}
</style> 