-- phpMyAdmin SQL Dump
-- version 4.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Sep 14, 2016 at 04:10 
-- Server version: 10.1.13-MariaDB
-- PHP Version: 5.6.20

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `museum`
--

-- --------------------------------------------------------

--
-- Table structure for table `cities`
--

CREATE TABLE `cities` (
  `id` int(10) UNSIGNED NOT NULL,
  `user_id` int(10) UNSIGNED NOT NULL,
  `country_id` int(10) UNSIGNED NOT NULL,
  `name` varchar(256) COLLATE utf8_unicode_ci NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `updated_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Dumping data for table `cities`
--

INSERT INTO `cities` (`id`, `user_id`, `country_id`, `name`, `created_at`, `updated_at`) VALUES
(1, 2, 1, 'مشهد', '2016-09-10 04:44:41', '2016-09-10 04:44:41'),
(2, 2, 1, 'تهران', '2016-09-10 04:44:52', '2016-09-10 04:44:52'),
(3, 2, 1, 'اصفهان', '2016-09-10 04:45:04', '2016-09-10 04:45:04'),
(4, 2, 1, 'تبریز', '2016-09-10 04:45:15', '2016-09-10 04:45:15'),
(5, 2, 3, 'روم', '2016-09-10 04:50:07', '2016-09-10 04:50:07'),
(6, 2, 2, 'پاریس', '2016-09-10 04:50:18', '2016-09-10 04:50:18'),
(7, 2, 4, 'سانفرانسیسکو', '2016-09-10 04:50:31', '2016-09-10 04:50:31'),
(8, 2, 4, 'لوس آنجلس', '2016-09-10 04:50:49', '2016-09-10 04:50:49'),
(9, 2, 5, 'توکیو', '2016-09-10 04:51:18', '2016-09-10 04:51:18'),
(10, 2, 6, 'پرتوریا', '2016-09-10 04:52:49', '2016-09-10 04:52:49');

-- --------------------------------------------------------

--
-- Table structure for table `countries`
--

CREATE TABLE `countries` (
  `id` int(10) UNSIGNED NOT NULL,
  `user_id` int(10) UNSIGNED NOT NULL,
  `name` varchar(256) COLLATE utf8_unicode_ci NOT NULL,
  `continent` varchar(16) COLLATE utf8_unicode_ci NOT NULL,
  `timezone` varchar(8) COLLATE utf8_unicode_ci NOT NULL DEFAULT '0',
  `language` varchar(255) COLLATE utf8_unicode_ci DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `updated_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Dumping data for table `countries`
--

INSERT INTO `countries` (`id`, `user_id`, `name`, `continent`, `timezone`, `language`, `created_at`, `updated_at`) VALUES
(1, 2, 'ایران', 'asia', '+4.5', 'فارسی', '2016-09-10 04:07:19', '2016-09-10 04:10:37'),
(2, 2, 'فرانسه', 'europe', '+1', 'فرانسوی', '2016-09-10 04:45:57', '2016-09-10 04:45:57'),
(3, 2, 'ایتالیا', 'europe', '+1', 'ایتالیایی', '2016-09-10 04:46:34', '2016-09-10 04:46:34'),
(4, 2, 'آمریکا', 'america', '-6', 'انگلیسی', '2016-09-10 04:47:21', '2016-09-10 04:48:12'),
(5, 2, 'ژاپن', 'asia', '+9', 'ژاپنی', '2016-09-10 04:47:57', '2016-09-10 04:48:46'),
(6, 2, 'آفریقا جنوبی', 'africa', '+2', 'آفریقایی - زولو', '2016-09-10 04:49:45', '2016-09-10 04:49:45');

-- --------------------------------------------------------

--
-- Table structure for table `historical_periods`
--

CREATE TABLE `historical_periods` (
  `id` int(10) UNSIGNED NOT NULL,
  `user_id` int(10) UNSIGNED NOT NULL,
  `name` varchar(256) COLLATE utf8_unicode_ci NOT NULL,
  `begin_year` int(11) NOT NULL DEFAULT '0',
  `end_year` int(11) NOT NULL DEFAULT '0',
  `created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `updated_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Dumping data for table `historical_periods`
--

INSERT INTO `historical_periods` (`id`, `user_id`, `name`, `begin_year`, `end_year`, `created_at`, `updated_at`) VALUES
(2, 2, 'هخامنشیان', 1200, 1250, '2016-09-09 19:30:00', '2016-09-10 02:29:10'),
(3, 2, 'ساسانیان', 1100, 1150, '2016-09-10 01:00:02', '2016-09-10 01:00:02'),
(4, 2, 'اشکانیان', 1000, 1050, '2016-09-10 02:45:43', '2016-09-10 02:45:43'),
(5, 2, 'دوره مفرغ', 800, 880, '2016-09-10 13:08:56', '2016-09-10 13:08:56'),
(6, 2, 'قاجاریه', 1050, 1100, '2016-09-10 13:41:56', '2016-09-10 13:41:56');

-- --------------------------------------------------------

--
-- Table structure for table `items`
--

CREATE TABLE `items` (
  `id` int(10) UNSIGNED NOT NULL,
  `user_id` int(10) UNSIGNED NOT NULL,
  `museum_id` int(10) UNSIGNED NOT NULL,
  `historical_period_id` int(10) UNSIGNED NOT NULL,
  `name` varchar(256) COLLATE utf8_unicode_ci NOT NULL,
  `count` int(10) UNSIGNED NOT NULL DEFAULT '1',
  `weight` varchar(16) COLLATE utf8_unicode_ci DEFAULT NULL,
  `dimentions` varchar(32) COLLATE utf8_unicode_ci DEFAULT NULL,
  `material` varchar(32) COLLATE utf8_unicode_ci DEFAULT NULL,
  `age` varchar(32) COLLATE utf8_unicode_ci NOT NULL,
  `discovery_site` varchar(128) COLLATE utf8_unicode_ci DEFAULT NULL,
  `image1` text COLLATE utf8_unicode_ci,
  `image2` text COLLATE utf8_unicode_ci,
  `image3` text COLLATE utf8_unicode_ci,
  `image4` text COLLATE utf8_unicode_ci,
  `created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `updated_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Dumping data for table `items`
--

INSERT INTO `items` (`id`, `user_id`, `museum_id`, `historical_period_id`, `name`, `count`, `weight`, `dimentions`, `material`, `age`, `discovery_site`, `image1`, `image2`, `image3`, `image4`, `created_at`, `updated_at`) VALUES
(1, 2, 1, 2, 'سردیس شاهزاده هخامنشی', 1, '3200', '10x12x18', 'سنگ', '1900', 'تخت جمشید', '/images/items/image11473529316.jpg', '/images/items/image21473529372.png', NULL, NULL, '2016-09-10 09:52:10', '2016-09-13 02:41:06'),
(2, 2, 1, 5, 'ظرف سنگی', 1, '', '', 'سنگ', '2800', 'جیرفت کرمان', '/images/items/image11473529293.jpg', NULL, NULL, NULL, '2016-09-10 13:11:33', '2016-09-13 02:46:28'),
(3, 2, 2, 6, 'سلاح های دوره قاجار', 3, '', '', 'چوب و فلز', '1800', 'مخروبه های اصفهان', '/images/items/image11473531095.jpg', NULL, NULL, NULL, '2016-09-10 13:41:35', '2016-09-13 02:52:56');

-- --------------------------------------------------------

--
-- Table structure for table `migrations`
--

CREATE TABLE `migrations` (
  `migration` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `batch` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Dumping data for table `migrations`
--

INSERT INTO `migrations` (`migration`, `batch`) VALUES
('2014_10_12_000000_create_users_table', 1),
('2014_10_12_100000_create_password_resets_table', 1),
('2016_09_09_121633_create_museums_table', 1),
('2016_09_09_121736_create_items_table', 1),
('2016_09_09_121753_create_musems_items_table', 1),
('2016_09_09_121816_create_countries_table', 1),
('2016_09_09_121903_create_cities_table', 1),
('2016_09_09_121936_create_historical_periods_table', 1);

-- --------------------------------------------------------

--
-- Table structure for table `museums`
--

CREATE TABLE `museums` (
  `id` int(10) UNSIGNED NOT NULL,
  `user_id` int(10) UNSIGNED NOT NULL,
  `city_id` int(10) UNSIGNED NOT NULL,
  `name` varchar(256) COLLATE utf8_unicode_ci NOT NULL,
  `area` int(10) UNSIGNED NOT NULL DEFAULT '0',
  `num_of_halls` tinyint(3) UNSIGNED NOT NULL DEFAULT '0',
  `phone` varchar(64) COLLATE utf8_unicode_ci DEFAULT NULL,
  `email` varchar(64) COLLATE utf8_unicode_ci DEFAULT NULL,
  `address` varchar(256) COLLATE utf8_unicode_ci DEFAULT NULL,
  `images` text COLLATE utf8_unicode_ci,
  `created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `updated_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Dumping data for table `museums`
--

INSERT INTO `museums` (`id`, `user_id`, `city_id`, `name`, `area`, `num_of_halls`, `phone`, `email`, `address`, `images`, `created_at`, `updated_at`) VALUES
(1, 2, 2, 'موزه ملی ایران', 18000, 24, '02166702061', 'info@nmi.ichto.ir', 'میدان امام خمینی، خیابان امام خمینی ابتدای خیابان سی تیر خیابان پروفسور رولن شماره یک', NULL, '2016-09-10 09:35:10', '2016-09-10 09:35:10'),
(2, 2, 4, 'موزه قاجار', 1500, 6, '04185698545', 'info@ghajarmuseum.com', 'خیابان سردار شهید مظفری - پلاک 143', NULL, '2016-09-10 13:38:32', '2016-09-10 13:38:32');

-- --------------------------------------------------------

--
-- Table structure for table `museums_items`
--

CREATE TABLE `museums_items` (
  `id` int(10) UNSIGNED NOT NULL,
  `user_id` int(10) UNSIGNED NOT NULL,
  `museum_id` int(10) UNSIGNED NOT NULL,
  `item_id` int(10) UNSIGNED NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `updated_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `password_resets`
--

CREATE TABLE `password_resets` (
  `email` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `token` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(10) UNSIGNED NOT NULL,
  `name` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `lname` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `email` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `password` varchar(60) COLLATE utf8_unicode_ci NOT NULL,
  `group` varchar(32) COLLATE utf8_unicode_ci NOT NULL,
  `remember_token` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `updated_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `name`, `lname`, `email`, `password`, `group`, `remember_token`, `created_at`, `updated_at`) VALUES
(2, 'مهران', 'احمدیان', 'info@ahmadian.me', '$2y$10$gZELQGOkyMRM1igMUCLQU.FnoY2x1pQT5/hElKVutmBBiBUcoWGii', '', 'IffQ0DxpKnAqW1pENJi9A5upIVHPrhEZx13UIaGwCRxBIAlkGDNOOT4374Ui', '2016-09-10 01:34:40', '2016-09-10 15:55:42'),
(3, 'کاربر', 'تست', 'test@test.tld', '$2y$10$x6OPOwC/IbUcsT6W50XiruXKYiHaJNVhjcdUutzy0/nnc7x2J8bPa', '', 'Dmr9KZG6Fr6AF8mGADAPutkfwZAWKyGkBaWLUtbKhP02Yi3tpZxaTXGXaoss', '2016-09-10 14:54:15', '2016-09-13 02:55:52');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `cities`
--
ALTER TABLE `cities`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `countries`
--
ALTER TABLE `countries`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `historical_periods`
--
ALTER TABLE `historical_periods`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `items`
--
ALTER TABLE `items`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `museums`
--
ALTER TABLE `museums`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `museums_items`
--
ALTER TABLE `museums_items`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `password_resets`
--
ALTER TABLE `password_resets`
  ADD KEY `password_resets_email_index` (`email`),
  ADD KEY `password_resets_token_index` (`token`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `users_email_unique` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `cities`
--
ALTER TABLE `cities`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;
--
-- AUTO_INCREMENT for table `countries`
--
ALTER TABLE `countries`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;
--
-- AUTO_INCREMENT for table `historical_periods`
--
ALTER TABLE `historical_periods`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;
--
-- AUTO_INCREMENT for table `items`
--
ALTER TABLE `items`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
--
-- AUTO_INCREMENT for table `museums`
--
ALTER TABLE `museums`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;
--
-- AUTO_INCREMENT for table `museums_items`
--
ALTER TABLE `museums_items`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT;
--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
